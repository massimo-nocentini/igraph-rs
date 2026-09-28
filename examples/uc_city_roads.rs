//! Use case: routes and capacity of a city road network.
//!
//! ```sh
//! cargo run --example uc_city_roads
//! ```
//!
//! The city is a grid of crossings; every road gets a random (seeded)
//! travel time and a random capacity. We compute the fastest route between
//! two corners with Dijkstra's algorithm, then the maximum traffic that can
//! flow from the west district to the east district, and the bottleneck
//! roads (a minimum cut) that limit it: by the max-flow min-cut theorem the
//! two numbers coincide.

use igraph::prelude::*;

const W: i64 = 8;
const H: i64 = 6;

fn crossing(v: i64) -> String {
    if v >= W * H {
        return if v == W * H { "WEST" } else { "EAST" }.to_string();
    }
    format!("({},{})", v / W, v % W)
}

fn main() -> igraph::Result<()> {
    // The grid: crossing (r, c) is vertex r * W + c.
    let mut edges = Vec::new();
    for r in 0..H {
        for c in 0..W - 1 {
            edges.push((r * W + c, r * W + c + 1));
        }
    }
    for r in 0..H - 1 {
        for c in 0..W {
            edges.push((r * W + c, (r + 1) * W + c));
        }
    }
    let city = Graph::from_edges(&edges, (W * H) as usize, false)?;
    rng::seed(2024)?;
    let minutes: Vec<f64> = (0..city.ecount())
        .map(|_| rng::integer(1, 9) as f64)
        .collect();
    println!(
        "City: {W} x {H} crossings, {} roads (1-9 minutes each)",
        city.ecount()
    );

    // --- Fastest route ----------------------------------------------------
    let (home, office) = (0, W * H - 1);
    let route = city.get_shortest_path_dijkstra(home, office, Some(&minutes), NeighborMode::All)?;
    let total: f64 = route.edges.iter().map(|&e| minutes[e as usize]).sum();
    let steps: Vec<String> = route.vertices.iter().map(|&v| crossing(v)).collect();
    println!(
        "\nFastest route {} -> {}: {total} minutes, {} roads",
        crossing(home),
        crossing(office),
        route.edges.len()
    );
    println!("  {}", steps.join(" -> "));
    let dist =
        city.distances_dijkstra(home, VertexSelector::All, Some(&minutes), NeighborMode::All)?;
    println!("\nTravel times from {} to every crossing:", crossing(home));
    for r in 0..H {
        let row: Vec<String> = (0..W)
            .map(|c| format!("{:>3}", dist[(0, (r * W + c) as usize)]))
            .collect();
        println!("  {}", row.join(""));
    }

    // --- District capacity ------------------------------------------------
    let capacity: Vec<f64> = (0..city.ecount())
        .map(|_| rng::integer(1, 5) as f64 * 100.0)
        .collect();
    let (west, east) = (W * H, W * H + 1);
    let mut net = city.clone();
    net.add_vertices(2)?;
    let mut caps = capacity.clone();
    for r in 0..H {
        net.add_edges(&[(west, r * W), (r * W + W - 1, east)])?;
        caps.extend([1e9, 1e9]); // district access roads: unlimited
    }
    let flow = net.maxflow(west, east, Some(&caps))?;
    let cut = net.st_mincut(west, east, Some(&caps))?;
    println!("\nWest -> east capacity (vehicles / hour):");
    println!("  maximum flow   = {}", flow.value);
    println!("  minimum cut    = {}", cut.value);
    println!("  bottleneck roads ({}):", cut.cut.len());
    for &e in &cut.cut {
        let (u, v) = net.edge(e)?;
        println!(
            "    {} -- {}  capacity {}",
            crossing(u),
            crossing(v),
            caps[e as usize]
        );
    }
    let busiest = (0..city.ecount())
        .max_by(|&a, &b| flow.flow[a].abs().total_cmp(&flow.flow[b].abs()))
        .unwrap_or(0);
    let (u, v) = net.edge(busiest as i64)?;
    println!(
        "  busiest road: {} -- {} with {} of {} vehicles / hour",
        crossing(u),
        crossing(v),
        flow.flow[busiest].abs(),
        caps[busiest]
    );
    Ok(())
}

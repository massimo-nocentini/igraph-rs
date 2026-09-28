//! Solving a maze with igraph's path algorithms.
//!
//! Run with `cargo run --example paths_maze`.
//!
//! The maze is a character grid: `#` are walls, `S` the start and `E` the
//! exit. Free cells become vertices, adjacent free cells are joined by an
//! edge. We then compare BFS, A* (with the Manhattan heuristic) and count how
//! many equally short solutions exist, then check a distance cutoff against
//! a bounded neighborhood from the `components` module.

use igraph::prelude::*;

const MAZE: [&str; 9] = [
    "###########",
    "#S  #     #",
    "# # # ### #",
    "# #   #   #",
    "# ##### # #",
    "#     # # #",
    "### # # # #",
    "#   #   #E#",
    "###########",
];

fn main() -> igraph::Result<()> {
    let width = MAZE[0].len() as i64;
    let cell = |x: i64, y: i64| MAZE[y as usize].as_bytes()[x as usize];
    let id = |x: i64, y: i64| x + width * y;
    let (mut start, mut exit) = (0, 0);
    let mut edges = vec![];
    for y in 0..MAZE.len() as i64 {
        for x in 0..width {
            match cell(x, y) {
                b'#' => continue,
                b'S' => start = id(x, y),
                b'E' => exit = id(x, y),
                _ => {}
            }
            if cell(x + 1, y) != b'#' {
                edges.push((id(x, y), id(x + 1, y)));
            }
            if cell(x, y + 1) != b'#' {
                edges.push((id(x, y), id(x, y + 1)));
            }
        }
    }
    let maze = Graph::from_edges(&edges, (width * MAZE.len() as i64) as usize, false)?;

    let bfs = maze.get_shortest_path(start, exit, None, NeighborMode::All)?;
    println!("BFS: the exit is {} steps away", bfs.len());

    let mut expanded = 0;
    let manhattan =
        |a: i64, b: i64| ((a % width - b % width).abs() + (a / width - b / width).abs()) as f64;
    let astar = maze.get_shortest_path_astar(start, exit, None, NeighborMode::All, |v, to| {
        expanded += 1;
        manhattan(v, to)
    })?;
    println!(
        "A*: {} steps, heuristic evaluated {expanded} times",
        astar.len()
    );
    assert_eq!(astar.len(), bfs.len());

    let all = maze.get_all_shortest_paths(start, exit, None, NeighborMode::All)?;
    println!("Number of shortest solutions: {}", all.nrgeo[exit as usize]);

    // Draw the first solution.
    let mut canvas: Vec<Vec<u8>> = MAZE.iter().map(|row| row.as_bytes().to_vec()).collect();
    for &v in &astar.vertices[1..astar.vertices.len() - 1] {
        canvas[(v / width) as usize][(v % width) as usize] = b'.';
    }
    for row in canvas {
        println!("{}", String::from_utf8_lossy(&row));
    }

    let far = maze.eccentricity(start, None, NeighborMode::All)?[0];
    println!("The farthest reachable cell from the start is {far} steps away");

    // Cells within 5 steps: a distance cutoff (paths) and a bounded
    // neighborhood (components) answer the same question.
    let near = maze.distances_cutoff(start, .., None, NeighborMode::All, Some(5.0))?;
    let within: Vec<i64> = maze
        .vertices()
        .filter(|&v| near[(0, v as usize)].is_finite())
        .collect();
    let mut ball = maze
        .neighborhood(start, Some(5), NeighborMode::All, 0)?
        .remove(0);
    ball.sort();
    assert_eq!(within, ball);
    println!("{} cells are at most 5 steps from the start", within.len());
    Ok(())
}

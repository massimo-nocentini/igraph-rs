//! Epidemics, spatial graphs, non-graph utilities and library-level hooks.
//!
//! This module gathers the parts of igraph that are not about one family of
//! graph algorithms, but are still very useful when working with networks:
//!
//! - **Epidemics** (`igraph_epidemics.h`): stochastic SIR simulations on a
//!   graph, see [`Graph::sir`] and [`SirRun`].
//! - **Spatial graphs** (`igraph_spatial.h`): graphs built from point clouds
//!   (Delaunay graph, Gabriel graph, relative neighborhood graph,
//!   β-skeletons, *k* nearest neighbor graphs), edge lengths from
//!   coordinates, and 2D convex hulls.
//! - **Non-graph utilities** (`igraph_nongraph.h`): power-law fitting
//!   ([`power_law_fit`], [`PowerLawFit`]), running means, sequential random
//!   sampling, tolerant floating-point comparisons.
//! - **Linear sum assignment** (`igraph_lsap.h`): the Hungarian method,
//!   [`solve_lsap`].
//! - **Random sampling of vectors** (`igraph_sampling.h`): uniform points on
//!   or inside a sphere, Dirichlet samples, both from the thread's default
//!   generator and from an explicit [`Rng`](crate::rng::Rng).
//! - **Partial prefix-sum trees** (`igraph_psumtree.h`): [`PsumTree`], a
//!   data structure to sample from a changing discrete distribution in
//!   `O(log n)`.
//! - **Library hooks** (`igraph_version.h`, `igraph_progress.h`,
//!   `igraph_statusbar.h`, `igraph_interrupt.h`): [`version`], and Rust
//!   closures installed as *progress*, *status* and *interruption*
//!   handlers of the calling thread, with RAII guards that restore the
//!   previous handler.
//!
//! # Example
//!
//! Build the Delaunay triangulation of the corners and the center of a
//! square, weight its edges by their Euclidean length, and solve a small
//! assignment problem:
//!
//! ```
//! use igraph::misc::{self, Metric};
//! use igraph::prelude::*;
//!
//! let points = Matrix::from_rows(&[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.5, 0.5]])?;
//! let delaunay = Graph::delaunay_graph(&points)?;
//! // 4 sides of the square + 4 spokes towards the center.
//! assert_eq!(delaunay.ecount(), 8);
//! let lengths = delaunay.spatial_edge_lengths(&points, Metric::Euclidean)?;
//! assert!(lengths.iter().all(|&l| l == 1.0 || (l - 0.5f64.sqrt()).abs() < 1e-12));
//!
//! // Three workers, three jobs: who does what at minimum total cost?
//! let cost = Matrix::from_rows(&[[4.0, 1.0, 3.0], [2.0, 0.0, 5.0], [3.0, 2.0, 2.0]])?;
//! assert_eq!(misc::solve_lsap(&cost)?, vec![1, 0, 2]);
//!
//! let v = misc::version();
//! assert_eq!(v.major, 1);
//! # Ok::<(), igraph::Error>(())
//! ```
//!
//! # Randomness
//!
//! [`Graph::sir`], [`random_sample`], [`PowerLawFit::p_value`], the
//! samplers and [`PsumTree::sample`] draw from the calling thread's default
//! generator: every thread has its own, so [`rng::seed`](crate::rng::seed)
//! makes a thread's results reproducible without affecting other threads.
//!
//! # Workarounds for igraph bugs
//!
//! Some inputs make igraph 1.0.0 and 1.0.1 misbehave (the relevant C
//! sources did not change in 1.0.1); the wrappers handle them on the Rust
//! side:
//!
//! - [`Graph::lune_beta_skeleton`] with `beta > 2` or `beta < 0.5`, and
//!   [`Graph::circle_beta_skeleton`] with `beta < 0.5`, where igraph returns
//!   spurious edges;
//! - [`solve_lsap`] with NaN or infinite costs, where igraph loops forever
//!   (rejected);
//! - [`random_sample`] with `length == 0` and `low == high`, where igraph
//!   returns `[low]`, and with `low == i64::MIN`, where igraph negates
//!   `low` (undefined behavior in C; the wrapper samples from a shifted
//!   interval);
//! - [`PowerLawFit::p_value`] on invalid models (empty sample, `alpha <= 1`
//!   or not finite, invalid `xmin`) or with a tiny precision, where igraph
//!   crashes, loops forever or overflows a C `long` (rejected).
//!
//! # See also
//!
//! - [`crate::games`] for random graph models, e.g.
//!   [`Graph::grg_game`](crate::Graph::grg_game) (random geometric graphs)
//!   and [`Graph::dot_product_game`](crate::Graph::dot_product_game) (with
//!   latent positions from [`sample_sphere_surface`]);
//! - [`crate::layout`] for point sets to feed to the spatial functions;
//! - [`crate::paths`] and [`crate::structural`] for weighted computations
//!   using [`Graph::spatial_edge_lengths`] as weights;
//! - [`crate::rng`] for the random number generators.
//!
//! # Contents
//!
//! | Rust | C function(s) | Chapter |
//! |------|---------------|---------|
//! | [`Graph::sir`], [`SirRun`] | `igraph_sir`, `igraph_sir_t` (`igraph_sir_init` / `igraph_sir_destroy` are memory plumbing, not wrapped: [`SirRun`] owns the vectors and frees them on drop) | Processes |
//! | [`Graph::delaunay_graph`] | `igraph_delaunay_graph` | Spatial |
//! | [`Graph::gabriel_graph`] | `igraph_gabriel_graph` | Spatial |
//! | [`Graph::relative_neighborhood_graph`] | `igraph_relative_neighborhood_graph` | Spatial |
//! | [`Graph::lune_beta_skeleton`] | `igraph_lune_beta_skeleton` | Spatial |
//! | [`Graph::circle_beta_skeleton`] | `igraph_circle_beta_skeleton` | Spatial |
//! | [`Graph::beta_weighted_gabriel_graph`] | `igraph_beta_weighted_gabriel_graph` | Spatial |
//! | [`Graph::nearest_neighbor_graph`] | `igraph_nearest_neighbor_graph` | Spatial |
//! | [`Graph::spatial_edge_lengths`] | `igraph_spatial_edge_lengths` | Spatial |
//! | [`convex_hull_2d`] | `igraph_convex_hull_2d` | Spatial |
//! | [`power_law_fit`], [`PowerLawFit::p_value`] | `igraph_power_law_fit`, `igraph_plfit_result_calculate_p_value` | Nongraph |
//! | [`running_mean`] | `igraph_running_mean` | Nongraph |
//! | [`random_sample`] | `igraph_random_sample` | Nongraph |
//! | [`almost_equals`], [`cmp_epsilon`] | `igraph_almost_equals`, `igraph_cmp_epsilon` | Nongraph |
//! | [`solve_lsap`] | `igraph_solve_lsap` | — |
//! | [`sample_sphere_surface`], [`sample_sphere_volume`], [`sample_dirichlet`] (and the [`Rng`](crate::rng::Rng) methods) | `igraph_rng_sample_*` | Nongraph |
//! | [`PsumTree`] | `igraph_psumtree_*` | Data structures |
//! | [`version`] | `igraph_version` | Nongraph |
//! | [`set_progress_handler`], [`with_progress_handler`], [`set_progress_handler_stderr`], [`progress`] | `igraph_set_progress_handler`, `igraph_progress_handler_stderr`, `igraph_progress` | Advanced |
//! | [`set_status_handler`], [`with_status_handler`], [`set_status_handler_stderr`], [`status`] | `igraph_set_status_handler`, `igraph_status_handler_stderr`, `igraph_status` | Advanced |
//! | [`set_interruption_handler`], [`with_interruption_handler`], [`allow_interruption`] | `igraph_set_interruption_handler`, `igraph_allow_interruption` | — |
//!
//! [`Graph::sir`]: crate::Graph::sir
//! [`Graph::delaunay_graph`]: crate::Graph::delaunay_graph
//! [`Graph::gabriel_graph`]: crate::Graph::gabriel_graph
//! [`Graph::relative_neighborhood_graph`]: crate::Graph::relative_neighborhood_graph
//! [`Graph::lune_beta_skeleton`]: crate::Graph::lune_beta_skeleton
//! [`Graph::circle_beta_skeleton`]: crate::Graph::circle_beta_skeleton
//! [`Graph::beta_weighted_gabriel_graph`]: crate::Graph::beta_weighted_gabriel_graph
//! [`Graph::nearest_neighbor_graph`]: crate::Graph::nearest_neighbor_graph
//! [`Graph::spatial_edge_lengths`]: crate::Graph::spatial_edge_lengths
//! [`PsumTree::sample`]: crate::misc::PsumTree::sample

mod epidemics;
mod handlers;
mod lsap;
mod nongraph;
mod psumtree;
mod sampling;
mod spatial;

pub use epidemics::SirRun;
pub use handlers::{
    InterruptionHandlerGuard, ProgressHandlerGuard, StatusHandlerGuard, allow_interruption,
    progress, set_interruption_handler, set_progress_handler, set_progress_handler_stderr,
    set_status_handler, set_status_handler_stderr, status, with_interruption_handler,
    with_progress_handler, with_status_handler,
};
pub use lsap::solve_lsap;
pub use nongraph::{
    PowerLawFit, Version, almost_equals, cmp_epsilon, power_law_fit, random_sample, running_mean,
    version,
};
pub use psumtree::PsumTree;
pub use sampling::{sample_dirichlet, sample_sphere_surface, sample_sphere_volume};
pub use spatial::{ConvexHull, Metric, convex_hull_2d};

/// Converts a nullable C string into an owned Rust string (lossy UTF-8).
fn lossy(ptr: *const std::ffi::c_char) -> String {
    if ptr.is_null() {
        String::new()
    } else {
        unsafe { std::ffi::CStr::from_ptr(ptr) }
            .to_string_lossy()
            .into_owned()
    }
}

/// Splits a column-major `nrow × ncol` matrix into its columns.
fn columns_of(m: &crate::matrix::Matrix) -> Vec<Vec<f64>> {
    m.columns().map(<[f64]>::to_vec).collect()
}

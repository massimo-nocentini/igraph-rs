//! Spatial graphs and computational geometry (`igraph_spatial.h`).
//!
//! Point sets are given as a [`Matrix`] with one point per row; the number
//! of columns is the dimension of the space. All the graph builders except
//! [`convex_hull_2d`] are marked *experimental* in igraph 1.0.x.
//!
//! The lune- and circle-based β-skeletons of igraph 1.0.0 and 1.0.1 return
//! spurious edges for some ranges of β; the wrappers of this module detect
//! these ranges and compute the correct skeletons instead (see
//! [`Graph::lune_beta_skeleton`]).
//!
//! See also [`Graph::grg_game`] (random geometric graphs, which are
//! nearest neighbor graphs with a distance cutoff), the layouts of
//! [`crate::layout`] (which produce point sets for these functions), and
//! the weighted path functions of [`crate::paths`], which take the output of
//! [`Graph::spatial_edge_lengths`] as weights.

use crate::{
    error::{Error, Result},
    ffi::*,
    graph::Graph,
    igraph_call,
    matrix::Matrix,
    vector::{Vector, VectorInt},
};

crate::ffi_enum! {
    /// The distance metric used by spatial functions (`igraph_metric_t`).
    pub enum Metric: igraph_metric_t {
        /// The Euclidean (L2) distance, `sqrt(Σ (x_i - y_i)²)`.
        Euclidean = igraph_metric_t_IGRAPH_METRIC_EUCLIDEAN,
        /// The Manhattan (L1, taxicab) distance, `Σ |x_i - y_i|`.
        Manhattan = igraph_metric_t_IGRAPH_METRIC_MANHATTAN,
    }
}

impl Metric {
    /// Alias of [`Metric::Euclidean`] (`IGRAPH_METRIC_L2`).
    pub const L2: Metric = Metric::Euclidean;
    /// Alias of [`Metric::Manhattan`] (`IGRAPH_METRIC_L1`).
    pub const L1: Metric = Metric::Manhattan;
}

impl igraph_t {
    /// The Delaunay graph of a point set: two points are adjacent when they
    /// share an edge of the Delaunay triangulation (tetrahedralization, ...)
    /// of the points.
    ///
    /// `points` has one point per row, in any dimension `d >= 1`; vertex `i`
    /// of the result is the point in row `i`. The Delaunay graph is a
    /// supergraph of the [Gabriel graph](Graph::gabriel_graph), itself a
    /// supergraph of the [relative neighborhood graph](Graph::relative_neighborhood_graph)
    /// and of the Euclidean minimum spanning tree. The computation relies on
    /// Qhull.
    ///
    /// See also [`convex_hull_2d`]: in 2D, the outer boundary of the
    /// triangulation is the convex hull of the points.
    ///
    /// Binds [`igraph_delaunay_graph`](https://igraph.org/c/html/latest/igraph-Spatial.html#igraph_delaunay_graph)
    /// (experimental). Time complexity: `O(n log n)` for `d <= 3`, and
    /// `O(n^⌊d/2⌋ / ⌊d/2⌋!)` in general.
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) for
    /// duplicate points, non-finite coordinates, zero-dimensional points,
    /// and (currently) for degenerate sets that do not span the space, such
    /// as `d + 1` or more points all lying on a hyperplane.
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    /// // A 3x3 square lattice (igraph's own test case).
    /// let pts: Vec<[f64; 2]> =
    ///     (0..9).map(|i| [(i / 3) as f64, (i % 3) as f64]).collect();
    /// let g = Graph::delaunay_graph(&Matrix::from_rows(&pts)?)?;
    /// assert_eq!(g.vcount(), 9);
    /// // 12 lattice sides plus one diagonal in each of the 4 cells.
    /// assert_eq!(g.ecount(), 16);
    /// # Ok::<(), igraph::Error>(())
    /// ```
    pub fn delaunay_graph(points: &Matrix) -> Result<Graph> {
        Graph::init_with(|g| unsafe { igraph_delaunay_graph(g, points) })
    }

    /// The lune-based β-skeleton of a point set.
    ///
    /// Two points `A` and `B` are adjacent when no other point lies in their
    /// (closed) *lune*, a region whose size grows with `beta`: larger values
    /// of `beta` give sparser graphs. For `beta >= 1` the lune is the
    /// intersection of the two balls of radius `beta·|AB|/2` centered on the
    /// line `AB` and passing through `A` and `B` respectively; for
    /// `beta < 1` it is the intersection of the two disks of radius
    /// `|AB|/(2·beta)` whose boundaries pass through both `A` and `B`.
    /// `beta = 1` gives the [Gabriel graph](Graph::gabriel_graph);
    /// `beta = 2` is (almost, see
    /// [`relative_neighborhood_graph`](Graph::relative_neighborhood_graph))
    /// the relative neighborhood graph. Values of `beta < 1` are only
    /// supported in 2D, and are considerably slower.
    ///
    /// # Correctness workarounds
    ///
    /// igraph 1.0.0 and 1.0.1 under-estimate the search radius of the lune for
    /// `beta > 2` and `beta < 0.5`, and then returns spurious edges (for
    /// `beta < 0.5`, even the complete graph). This wrapper returns the
    /// correct skeleton in these ranges: for `beta > 2` it keeps the edges of
    /// [`beta_weighted_gabriel_graph`](Graph::beta_weighted_gabriel_graph)
    /// whose threshold exceeds `beta` (same complexity; point sets with no
    /// more points than dimensions are tested pair by pair, as igraph does),
    /// for `beta < 0.5` it tests every pair of points against every other
    /// point (`O(n³)`).
    ///
    /// Binds [`igraph_lune_beta_skeleton`](https://igraph.org/c/html/latest/igraph-Spatial.html#igraph_lune_beta_skeleton)
    /// (experimental). Time complexity: about `O(n^⌊d/2⌋ log n)`.
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) unless
    /// `beta` is positive and finite, or for NaN or infinite coordinates;
    /// [`ErrorKind::Unimplemented`](crate::ErrorKind::Unimplemented)
    /// for `beta < 1` outside of 2D; and, for `beta >= 1` with more points
    /// than dimensions (where the candidate edges come from the Delaunay
    /// graph), the errors of [`delaunay_graph`](Graph::delaunay_graph), e.g.
    /// for duplicate points.
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    /// // Two points and a third one slightly off their midpoint.
    /// let pts = Matrix::from_rows(&[[0.0, 0.0], [2.0, 0.0], [1.0, 1.2]])?;
    /// // For beta = 1 the third point is outside the disk with diameter 0-1...
    /// assert!(Graph::lune_beta_skeleton(&pts, 1.0)?.get_eid(0, 1, false)?.is_some());
    /// // ...but it is inside the fatter lune of beta = 2.
    /// assert!(Graph::lune_beta_skeleton(&pts, 2.0)?.get_eid(0, 1, false)?.is_none());
    /// # Ok::<(), igraph::Error>(())
    /// ```
    pub fn lune_beta_skeleton(points: &Matrix, beta: f64) -> Result<Graph> {
        check_beta(beta)?;
        check_finite(points)?;
        let (n, dim) = (points.nrow(), points.ncol());
        if beta > 2.0 {
            if n <= dim {
                // Too few points for a Delaunay triangulation: igraph tests
                // every pair, so do we (there are at most `dim` points).
                return brute_force_skeleton(points, |a, b, p| in_lune(a, b, p, beta));
            }
            // Thresholds up to (just above) beta are enough to decide.
            let (g, thresholds) = Graph::beta_weighted_gabriel_graph(points, beta * (1.0 + 1e-9))?;
            let kept: Vec<(igraph_int_t, igraph_int_t)> = g
                .edge_list()
                .into_iter()
                .zip(thresholds)
                .filter(|&(_, t)| t > beta)
                .map(|(e, _)| e)
                .collect();
            return Graph::from_edges(&kept, n, false);
        }
        if beta < 0.5 && dim == 2 {
            return brute_force_skeleton(points, |a, b, p| in_lens(a, b, p, beta));
        }
        Graph::init_with(|g| unsafe { igraph_lune_beta_skeleton(g, points, beta) })
    }

    /// The circle-based β-skeleton of a 2D point set.
    ///
    /// For `beta >= 1`, `A` and `B` are adjacent when no other point lies in
    /// the *union* of the two disks of radius `beta·|AB|/2` whose boundaries
    /// pass through both `A` and `B`; for `beta < 1` the forbidden region is
    /// the intersection of the disks of radius `|AB|/(2·beta)`, as for the
    /// [lune-based skeleton](Graph::lune_beta_skeleton). `beta` must be
    /// positive; larger values give sparser graphs, and values below 1 are
    /// considerably slower. For `beta = 1` it coincides with the Gabriel
    /// graph.
    ///
    /// For `beta < 0.5` igraph 1.0.0 and 1.0.1 return spurious edges (see the
    /// [lune-based skeleton](Graph::lune_beta_skeleton)): this wrapper then
    /// computes the correct skeleton by brute force, in `O(n³)`.
    ///
    /// Binds [`igraph_circle_beta_skeleton`](https://igraph.org/c/html/latest/igraph-Spatial.html#igraph_circle_beta_skeleton)
    /// (experimental).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) unless
    /// `beta` is positive and finite, or for NaN or infinite coordinates;
    /// [`ErrorKind::Unimplemented`](crate::ErrorKind::Unimplemented)
    /// if the points are not two-dimensional.
    pub fn circle_beta_skeleton(points: &Matrix, beta: f64) -> Result<Graph> {
        check_beta(beta)?;
        check_finite(points)?;
        if beta < 0.5 && points.ncol() == 2 {
            return brute_force_skeleton(points, |a, b, p| in_lens(a, b, p, beta));
        }
        Graph::init_with(|g| unsafe { igraph_circle_beta_skeleton(g, points, beta) })
    }

    /// The Gabriel graph together with, for each edge, the threshold β at
    /// which the edge disappears from the lune-based β-skeleton.
    ///
    /// The edge `e` belongs to the [lune β-skeleton](Graph::lune_beta_skeleton)
    /// exactly for `1 <= β < weights[e]`, so this single call summarizes all
    /// the skeletons with `β >= 1`. Edges that persist for arbitrarily large
    /// β, or beyond the `max_beta` cutoff, get the weight
    /// [`f64::INFINITY`]. A smaller `max_beta` makes the computation faster;
    /// pass [`f64::INFINITY`] for no cutoff.
    ///
    /// Returns the graph and the weights, indexed by edge id.
    ///
    /// Binds [`igraph_beta_weighted_gabriel_graph`](https://igraph.org/c/html/latest/igraph-Spatial.html#igraph_beta_weighted_gabriel_graph)
    /// (experimental).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) if
    /// `max_beta` is NaN, and the errors of
    /// [`delaunay_graph`](Graph::delaunay_graph) (which it uses even for
    /// tiny point sets: it needs more points than dimensions, and there must
    /// be no duplicate points).
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    /// let pts = Matrix::from_rows(&[[0.0, 0.0], [2.0, 0.0], [1.0, 1.2]])?;
    /// let (g, beta) = Graph::beta_weighted_gabriel_graph(&pts, f64::INFINITY)?;
    /// assert_eq!(g.ecount(), 3);
    /// let long = g.get_eid(0, 1, false)?.unwrap() as usize;
    /// // Edge 0-1 leaves the lune skeleton at beta = 1.22 (the third point
    /// // enters its lune), the other two sides at a much larger beta.
    /// assert!((beta[long] - 1.22).abs() < 1e-9);
    /// assert!(beta.iter().all(|&b| b >= beta[long]));
    /// # Ok::<(), igraph::Error>(())
    /// ```
    pub fn beta_weighted_gabriel_graph(
        points: &Matrix,
        max_beta: f64,
    ) -> Result<(Graph, Vec<f64>)> {
        if max_beta.is_nan() {
            return Err(Error::invalid("max_beta must not be NaN"));
        }
        let mut weights = Vector::new();
        let g = Graph::init_with(|g| unsafe {
            igraph_beta_weighted_gabriel_graph(g, &mut weights, points, max_beta)
        })?;
        Ok((g, weights.into()))
    }

    /// The Gabriel graph of a point set: `A` and `B` are adjacent when no
    /// other point lies in the closed ball having the segment `AB` as a
    /// diameter.
    ///
    /// The Gabriel graph is connected, planar in 2D, and it is the β-skeleton
    /// (lune- or circle-based) with `β = 1`. Any dimension is supported.
    ///
    /// Binds [`igraph_gabriel_graph`](https://igraph.org/c/html/latest/igraph-Spatial.html#igraph_gabriel_graph)
    /// (experimental). Time complexity: about `O(n^⌊d/2⌋ log n)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    /// // An obtuse triangle: the long side has the third point inside its
    /// // diametral circle, so it is not a Gabriel edge.
    /// let pts = Matrix::from_rows(&[[0.0, 0.0], [4.0, 0.0], [2.0, 0.5]])?;
    /// let g = Graph::gabriel_graph(&pts)?;
    /// assert_eq!(g.ecount(), 2);
    /// assert_eq!(g.get_eid(0, 1, false)?, None);
    /// # Ok::<(), igraph::Error>(())
    /// ```
    pub fn gabriel_graph(points: &Matrix) -> Result<Graph> {
        Graph::init_with(|g| unsafe { igraph_gabriel_graph(g, points) })
    }

    /// The relative neighborhood graph of a point set: `A` and `B` are
    /// adjacent unless some other point `C` is strictly closer to both of
    /// them than they are to each other (`AC < AB` and `BC < AB`).
    ///
    /// It is always connected, and it is a supergraph of the Euclidean
    /// minimum spanning tree. Unlike the `β = 2` lune skeleton (which uses
    /// non-strict inequalities and is triangle-free), it connects the three
    /// corners of an equilateral triangle.
    ///
    /// Binds [`igraph_relative_neighborhood_graph`](https://igraph.org/c/html/latest/igraph-Spatial.html#igraph_relative_neighborhood_graph)
    /// (experimental). Time complexity: about `O(n^⌊d/2⌋ log n)`.
    ///
    /// See also [`Graph::minimum_spanning_tree`]: with the
    /// [edge lengths](Graph::spatial_edge_lengths) as weights, the minimum
    /// spanning tree of this graph is the Euclidean minimum spanning tree of
    /// the points.
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::prelude::*;
    /// // A 1x2 rectangle with its center: the center is closer to every
    /// // corner than the corners to each other along the long sides.
    /// let pts = Matrix::from_rows(&[[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0], [1.0, 0.5]])?;
    /// let g = Graph::relative_neighborhood_graph(&pts)?;
    /// let mut edges = g.edge_list();
    /// edges.sort();
    /// // The two short sides and the four spokes to the center.
    /// assert_eq!(edges, vec![(0, 3), (0, 4), (1, 2), (1, 4), (2, 4), (3, 4)]);
    /// # Ok::<(), igraph::Error>(())
    /// ```
    pub fn relative_neighborhood_graph(points: &Matrix) -> Result<Graph> {
        Graph::init_with(|g| unsafe { igraph_relative_neighborhood_graph(g, points) })
    }

    /// The *k* nearest neighbor graph of a point set.
    ///
    /// Each point is connected to (at most) its `k` nearest other points
    /// according to `metric`, considering only points closer than `cutoff`.
    /// `k = None` means no limit on the number of neighbors, and
    /// `cutoff = None` no limit on the distance (with both `None` the result
    /// is complete). With `directed`, the edge `i → j` means that `j` is
    /// among the neighbors of `i`; otherwise `i` and `j` are connected when
    /// *either* chose the other (mutual choices give a single edge). Ties
    /// between equidistant neighbors are broken arbitrarily.
    ///
    /// Binds [`igraph_nearest_neighbor_graph`](https://igraph.org/c/html/latest/igraph-Spatial.html#igraph_nearest_neighbor_graph)
    /// (experimental). Time complexity: `O(n log n)` (k-d tree).
    ///
    /// See also [`Graph::grg_game`], which samples random points in the unit
    /// square and connects those closer than a radius: the undirected graph
    /// built here with `k = None` and that radius as `cutoff`.
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) for
    /// zero-dimensional points, non-finite coordinates, or a negative or NaN
    /// `cutoff`.
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::{misc::Metric, prelude::*};
    /// // Points on a line: 0, 1, 3, 7.
    /// let pts = Matrix::from_rows(&[[0.0], [1.0], [3.0], [7.0]])?;
    /// let g = Graph::nearest_neighbor_graph(&pts, Metric::Euclidean, Some(1), None, true)?;
    /// let mut edges = g.edge_list();
    /// edges.sort();
    /// assert_eq!(edges, vec![(0, 1), (1, 0), (2, 1), (3, 2)]);
    /// # Ok::<(), igraph::Error>(())
    /// ```
    pub fn nearest_neighbor_graph(
        points: &Matrix,
        metric: Metric,
        k: Option<usize>,
        cutoff: Option<f64>,
        directed: bool,
    ) -> Result<Graph> {
        // More neighbors than points means no limit at all.
        let k = k.map_or(-1, |k| {
            igraph_int_t::try_from(k).unwrap_or(igraph_int_t::MAX)
        });
        let cutoff = match cutoff {
            None => f64::INFINITY,
            Some(c) if c >= 0.0 => c,
            Some(c) => {
                return Err(Error::invalid(format!(
                    "the cutoff distance must be non-negative, got {c}"
                )));
            }
        };
        Graph::init_with(|g| unsafe {
            igraph_nearest_neighbor_graph(g, points, metric.into(), k, cutoff, directed)
        })
    }

    /// The length of each edge, computed from the coordinates of its
    /// endpoints with the given `metric`, indexed by edge id.
    ///
    /// Row `i` of `points` holds the coordinates of vertex `i`, in any
    /// dimension. The lengths can be used as weights by path-length based
    /// functions, e.g. [`Graph::distances_dijkstra`],
    /// [`Graph::minimum_spanning_tree`], [`Graph::betweenness`],
    /// [`Graph::closeness`] or [`Graph::voronoi`].
    ///
    /// Binds [`igraph_spatial_edge_lengths`](https://igraph.org/c/html/latest/igraph-Spatial.html#igraph_spatial_edge_lengths)
    /// (experimental). Time complexity: `O(|E| d)`.
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) if the
    /// number of rows differs from the number of vertices, or the points are
    /// zero-dimensional (a `0 × 0` matrix is accepted for the null graph).
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::{misc::Metric, prelude::*};
    /// let g = Graph::from_edges(&[(0, 1), (0, 2)], 3, false)?;
    /// let pts = Matrix::from_rows(&[[0.0, 0.0], [3.0, 4.0], [1.0, 1.0]])?;
    /// assert_eq!(g.spatial_edge_lengths(&pts, Metric::Euclidean)?[0], 5.0);
    /// assert_eq!(g.spatial_edge_lengths(&pts, Metric::Manhattan)?, vec![7.0, 2.0]);
    /// # Ok::<(), igraph::Error>(())
    /// ```
    pub fn spatial_edge_lengths(&self, points: &Matrix, metric: Metric) -> Result<Vec<f64>> {
        let mut res = Vector::new();
        igraph_call!(igraph_spatial_edge_lengths(
            self,
            &mut res,
            points,
            metric.into()
        ))?;
        Ok(res.into())
    }
}

fn check_beta(beta: f64) -> Result<()> {
    if beta > 0.0 && beta.is_finite() {
        Ok(())
    } else {
        Err(Error::invalid(format!(
            "beta must be positive and finite, got {beta}"
        )))
    }
}

fn check_finite(points: &Matrix) -> Result<()> {
    if points.as_slice().iter().all(|x| x.is_finite()) {
        Ok(())
    } else {
        Err(Error::invalid("coordinates must not be NaN or infinite"))
    }
}

/// igraph's relative tolerance: the forbidden regions are closed.
const TOL: f64 = 1.0 + 128.0 * f64::EPSILON;

fn sqr_dist(p: &[f64], q: &[f64]) -> f64 {
    p.iter().zip(q).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// Whether `p` lies in the closed lune of the segment `ab` for `beta >= 1`:
/// the intersection of the balls of radius `beta·|ab|/2` centered at
/// `a + (beta/2 - 1)(a - b)` and `b + (beta/2 - 1)(b - a)` (any dimension).
fn in_lune(a: &[f64], b: &[f64], p: &[f64], beta: f64) -> bool {
    let r = beta / 2.0;
    let radius2 = r * r * sqr_dist(a, b) * TOL * TOL;
    let ca: Vec<f64> = a
        .iter()
        .zip(b)
        .map(|(x, y)| x + (r - 1.0) * (x - y))
        .collect();
    let cb: Vec<f64> = a
        .iter()
        .zip(b)
        .map(|(x, y)| y + (r - 1.0) * (y - x))
        .collect();
    sqr_dist(p, &ca) <= radius2 && sqr_dist(p, &cb) <= radius2
}

/// Whether the 2D point `p` lies in the closed lens of the segment `ab` for
/// `beta < 1`: the intersection of the two disks of radius `|ab|/(2 beta)`
/// whose boundaries pass through both `a` and `b`.
fn in_lens(a: &[f64], b: &[f64], p: &[f64], beta: f64) -> bool {
    let d2 = sqr_dist(a, b);
    let mid = [(a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0];
    // The lens lies within the disk having `ab` as a diameter.
    if sqr_dist(p, &mid) > d2 / 4.0 * TOL * TOL {
        return false;
    }
    let r = 0.5 / beta;
    let shift = (r * r - 0.25).sqrt();
    let perp = [-(a[1] - b[1]) * shift, (a[0] - b[0]) * shift];
    let c1 = [mid[0] + perp[0], mid[1] + perp[1]];
    let c2 = [mid[0] - perp[0], mid[1] - perp[1]];
    let radius2 = r * r * d2 * TOL * TOL;
    sqr_dist(p, &c1) <= radius2 && sqr_dist(p, &c2) <= radius2
}

/// A β-skeleton by brute force, in `O(n³ d)`: `a` and `b` are adjacent
/// unless `blocked(a, b, p)` holds for some other point `p`. Used where
/// igraph 1.0.0 and 1.0.1 are wrong.
fn brute_force_skeleton(
    points: &Matrix,
    blocked: impl Fn(&[f64], &[f64], &[f64]) -> bool,
) -> Result<Graph> {
    let n = points.nrow();
    let pts = points.to_rows();
    let mut edges = Vec::new();
    for a in 0..n {
        for b in a + 1..n {
            let hit = (0..n).any(|k| k != a && k != b && blocked(&pts[a], &pts[b], &pts[k]));
            if !hit {
                edges.push((a as igraph_int_t, b as igraph_int_t));
            }
        }
    }
    Graph::from_edges(&edges, n, false)
}

/// The convex hull of a 2D point set, as returned by [`convex_hull_2d`].
#[derive(Debug, Clone, PartialEq)]
pub struct ConvexHull {
    /// The indices (rows of the input matrix) of the corners of the hull,
    /// in order along the boundary. Points in the interior of a side are
    /// not corners.
    pub vertices: Vec<i64>,
    /// The coordinates of the corners, in the same order as `vertices`.
    pub points: Vec<[f64; 2]>,
}

impl ConvexHull {
    /// The (non-negative) area enclosed by the hull, by the shoelace formula.
    pub fn area(&self) -> f64 {
        let n = self.points.len();
        let twice: f64 = (0..n)
            .map(|i| {
                let [x1, y1] = self.points[i];
                let [x2, y2] = self.points[(i + 1) % n];
                x1 * y2 - x2 * y1
            })
            .sum();
        twice.abs() / 2.0
    }

    /// The perimeter of the hull.
    pub fn perimeter(&self) -> f64 {
        let n = self.points.len();
        if n < 2 {
            return 0.0;
        }
        (0..n)
            .map(|i| {
                let [x1, y1] = self.points[i];
                let [x2, y2] = self.points[(i + 1) % n];
                (x2 - x1).hypot(y2 - y1)
            })
            .sum()
    }
}

/// The convex hull of a set of points in the plane, by the Graham scan.
///
/// `points` must have two columns (x and y) and one point per row. The
/// corners are reported in order along the boundary; collinear points on
/// the sides are left out. Degenerate inputs are fine: one point gives a
/// one-corner hull, collinear points give the two extremes, no points an
/// empty hull.
///
/// Binds [`igraph_convex_hull_2d`](https://igraph.org/c/html/latest/igraph-Spatial.html#igraph_convex_hull_2d).
/// Time complexity: `O(n log n)`.
///
/// See also [`Graph::layout_circle`](crate::Graph::layout_circle) and the
/// other layouts of [`crate::layout`], whose coordinate matrices can be
/// passed directly.
///
/// # Errors
/// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) if `points`
/// does not have exactly two columns, or has NaN or infinite coordinates.
///
/// # Examples
///
/// ```
/// use igraph::{misc, prelude::*};
/// let pts = Matrix::from_rows(&[[0.0, 0.0], [2.0, 0.0], [1.0, 1.0], [2.0, 2.0], [0.0, 2.0]])?;
/// let hull = misc::convex_hull_2d(&pts)?;
/// let mut corners = hull.vertices.clone();
/// corners.sort();
/// assert_eq!(corners, vec![0, 1, 3, 4]); // the center point (row 2) is inside
/// assert_eq!(hull.area(), 4.0);
/// assert_eq!(hull.perimeter(), 8.0);
/// # Ok::<(), igraph::Error>(())
/// ```
pub fn convex_hull_2d(points: &Matrix) -> Result<ConvexHull> {
    check_finite(points)?;
    let mut vertices = VectorInt::new();
    let mut coords = Matrix::new();
    igraph_call!(igraph_convex_hull_2d(points, &mut vertices, &mut coords))?;
    let points = (0..coords.nrow())
        .map(|i| [coords[(i, 0)], coords[(i, 1)]])
        .collect();
    Ok(ConvexHull {
        vertices: vertices.into(),
        points,
    })
}

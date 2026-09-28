//! Sampling random vectors (`igraph_sampling.h`).
//!
//! Each sampler exists as a free function drawing from the thread's default
//! random number generator, and as a method of [`Rng`] drawing from an
//! explicit generator. Samples are returned as one `Vec<f64>` per point.

use super::columns_of;
use crate::{
    error::{Error, Result},
    ffi::*,
    igraph_call,
    matrix::Matrix,
    rng::Rng,
    vector::Vector,
};

fn default_rng() -> *mut igraph_rng_t {
    crate::error::ensure_init();
    unsafe { igraph_rng_default() }
}

fn to_int(x: usize, what: &str) -> Result<igraph_int_t> {
    igraph_int_t::try_from(x).map_err(|_| Error::invalid(format!("{what} is too large")))
}

fn check_radius(radius: f64) -> Result<()> {
    if radius > 0.0 && radius.is_finite() {
        Ok(())
    } else {
        Err(Error::invalid(format!(
            "the sphere radius must be positive and finite, got {radius}"
        )))
    }
}

fn sphere_surface(
    rng: *mut igraph_rng_t,
    dim: usize,
    n: usize,
    radius: f64,
    positive: bool,
) -> Result<Vec<Vec<f64>>> {
    check_radius(radius)?;
    let mut res = Matrix::new();
    igraph_call!(igraph_rng_sample_sphere_surface(
        rng,
        to_int(dim, "dimension")?,
        to_int(n, "number of samples")?,
        radius,
        positive,
        &mut res
    ))?;
    Ok(columns_of(&res))
}

fn sphere_volume(
    rng: *mut igraph_rng_t,
    dim: usize,
    n: usize,
    radius: f64,
    positive: bool,
) -> Result<Vec<Vec<f64>>> {
    check_radius(radius)?;
    let mut res = Matrix::new();
    igraph_call!(igraph_rng_sample_sphere_volume(
        rng,
        to_int(dim, "dimension")?,
        to_int(n, "number of samples")?,
        radius,
        positive,
        &mut res
    ))?;
    Ok(columns_of(&res))
}

fn dirichlet(rng: *mut igraph_rng_t, n: usize, alpha: &[f64]) -> Result<Vec<Vec<f64>>> {
    if !alpha.iter().all(|&a| a > 0.0 && a.is_finite()) {
        return Err(Error::invalid(
            "Dirichlet concentration parameters must be positive and finite",
        ));
    }
    let alpha = Vector::view(alpha);
    let mut res = Matrix::new();
    igraph_call!(igraph_rng_sample_dirichlet(
        rng,
        to_int(n, "number of samples")?,
        alpha.as_ptr(),
        &mut res
    ))?;
    Ok(columns_of(&res))
}

/// Samples `n` points uniformly from the surface of the `dim`-dimensional
/// sphere of the given `radius`, centered at the origin, using the thread's
/// default random number generator.
///
/// With `positive` set, the points are restricted to the positive orthant
/// (all coordinates non-negative). Each returned point has `dim`
/// coordinates and Euclidean norm `radius`. With `positive` and
/// `radius <= 1` they are valid latent positions for
/// [`Graph::dot_product_game`](crate::Graph::dot_product_game) (put them in
/// the *columns* of its matrix).
///
/// Binds [`igraph_rng_sample_sphere_surface`](https://igraph.org/c/html/latest/igraph-Nongraph.html#igraph_rng_sample_sphere_surface).
/// Time complexity: `O(n · dim)`.
///
/// # Errors
/// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) if
/// `dim < 2`, or `radius` is not positive and finite.
///
/// # Examples
///
/// ```
/// use igraph::{misc, prelude::*};
///
/// rng::seed(42)?;
/// for p in misc::sample_sphere_surface(3, 10, 2.0, true)? {
///     let norm = p.iter().map(|x| x * x).sum::<f64>().sqrt();
///     assert!((norm - 2.0).abs() < 1e-9);
///     assert!(p.iter().all(|&x| x >= 0.0));
/// }
/// # Ok::<(), igraph::Error>(())
/// ```
pub fn sample_sphere_surface(
    dim: usize,
    n: usize,
    radius: f64,
    positive: bool,
) -> Result<Vec<Vec<f64>>> {
    sphere_surface(default_rng(), dim, n, radius, positive)
}

/// Samples `n` points uniformly from the volume (the ball) of the
/// `dim`-dimensional sphere of the given `radius`, centered at the origin,
/// using the thread's default random number generator.
///
/// With `positive` set, the points are restricted to the positive orthant.
/// Each returned point has `dim` coordinates and Euclidean norm at most
/// `radius`.
///
/// See also [`Graph::grg_game`](crate::Graph::grg_game), which samples
/// points uniformly in the unit square and connects nearby ones, and
/// [`Graph::nearest_neighbor_graph`](crate::Graph::nearest_neighbor_graph)
/// to build a graph from sampled points in any dimension.
///
/// Binds [`igraph_rng_sample_sphere_volume`](https://igraph.org/c/html/latest/igraph-Nongraph.html#igraph_rng_sample_sphere_volume).
/// Time complexity: `O(n · dim)`.
///
/// # Errors
/// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) if
/// `dim < 2`, or `radius` is not positive and finite.
///
/// # Examples
///
/// ```
/// use igraph::{misc, prelude::*};
///
/// rng::seed(42)?;
/// let pts = misc::sample_sphere_volume(2, 4000, 1.0, false)?;
/// assert!(pts.iter().all(|p| p[0].hypot(p[1]) <= 1.0));
/// // Uniform in the disk: about a quarter of the points within radius 1/2.
/// let inner = pts.iter().filter(|p| p[0].hypot(p[1]) < 0.5).count() as f64 / 4000.0;
/// assert!((inner - 0.25).abs() < 0.03);
/// # Ok::<(), igraph::Error>(())
/// ```
pub fn sample_sphere_volume(
    dim: usize,
    n: usize,
    radius: f64,
    positive: bool,
) -> Result<Vec<Vec<f64>>> {
    sphere_volume(default_rng(), dim, n, radius, positive)
}

/// Samples `n` points from the Dirichlet distribution with concentration
/// parameters `alpha`, using the thread's default random number generator.
///
/// Each returned point has `alpha.len()` non-negative coordinates summing to
/// one (a random probability vector); its expected value is
/// `alpha / Σ alpha`, and larger concentrations give less spread-out samples.
/// Extremely small concentrations (e.g. `1e-300`) make igraph's gamma
/// samples underflow to zero, and then the coordinates are NaN.
///
/// Binds [`igraph_rng_sample_dirichlet`](https://igraph.org/c/html/latest/igraph-Nongraph.html#igraph_rng_sample_dirichlet).
/// Time complexity: `O(n · alpha.len())`.
///
/// # Errors
/// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) if `alpha`
/// has fewer than two entries or an entry that is not positive and finite.
///
/// # Examples
///
/// ```
/// use igraph::{misc, prelude::*};
///
/// rng::seed(42)?;
/// for p in misc::sample_dirichlet(100, &[1.0, 2.0, 3.0])? {
///     assert_eq!(p.len(), 3);
///     assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-9);
/// }
/// # Ok::<(), igraph::Error>(())
/// ```
pub fn sample_dirichlet(n: usize, alpha: &[f64]) -> Result<Vec<Vec<f64>>> {
    dirichlet(default_rng(), n, alpha)
}

impl Rng {
    /// Like [`sample_sphere_surface`],
    /// drawing from this generator (`igraph_rng_sample_sphere_surface`).
    ///
    /// ```
    /// use igraph::prelude::*;
    /// let mut a = Rng::new(RngType::Pcg64, 5)?;
    /// let mut b = Rng::new(RngType::Pcg64, 5)?;
    /// assert_eq!(a.sample_sphere_surface(4, 3, 1.0, false)?, b.sample_sphere_surface(4, 3, 1.0, false)?);
    /// # Ok::<(), igraph::Error>(())
    /// ```
    pub fn sample_sphere_surface(
        &mut self,
        dim: usize,
        n: usize,
        radius: f64,
        positive: bool,
    ) -> Result<Vec<Vec<f64>>> {
        sphere_surface(self, dim, n, radius, positive)
    }

    /// Like [`sample_sphere_volume`],
    /// drawing from this generator (`igraph_rng_sample_sphere_volume`).
    pub fn sample_sphere_volume(
        &mut self,
        dim: usize,
        n: usize,
        radius: f64,
        positive: bool,
    ) -> Result<Vec<Vec<f64>>> {
        sphere_volume(self, dim, n, radius, positive)
    }

    /// Like [`sample_dirichlet`], drawing from
    /// this generator (`igraph_rng_sample_dirichlet`).
    pub fn sample_dirichlet(&mut self, n: usize, alpha: &[f64]) -> Result<Vec<Vec<f64>>> {
        dirichlet(self, n, alpha)
    }
}

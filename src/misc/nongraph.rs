//! Non-graph utilities (`igraph_nongraph.h`, `igraph_version.h`).

use super::lossy;
use crate::{
    error::{Error, ErrorKind, Result},
    ffi::*,
    igraph_call,
    vector::{Vector, VectorInt},
};
use std::{cmp::Ordering, fmt};

/// Upper bound (2^62) on the samples (and `xmin`) of a discrete model
/// accepted by [`power_law_fit`] and [`PowerLawFit::p_value`].
const DISCRETE_SAMPLE_LIMIT: f64 = 4_611_686_018_427_387_904.0;

/// A power-law distribution fitted to a sample by [`power_law_fit`] (the
/// Rust counterpart of `igraph_plfit_result_t`).
///
/// The fitted model is `P(X = x) ∝ x^(-alpha)` for `x >= xmin`. The struct
/// borrows the fitted sample, which is needed to compute the
/// [p-value](PowerLawFit::p_value) of the fit.
#[derive(Debug, Clone, PartialEq)]
pub struct PowerLawFit<'a> {
    /// Whether a continuous (`true`) or a discrete (`false`) power law was fitted.
    pub continuous: bool,
    /// The fitted exponent `alpha` (larger than 1 for a normalizable law).
    pub alpha: f64,
    /// The threshold above which the power-law behavior holds (given, or
    /// estimated by minimizing the Kolmogorov–Smirnov statistic).
    pub xmin: f64,
    /// The log-likelihood of the fitted parameters (`L` in igraph).
    pub log_likelihood: f64,
    /// The Kolmogorov–Smirnov test statistic between the fitted
    /// distribution and the sample (`D` in igraph); the smaller, the better.
    pub ks_statistic: f64,
    /// The sample the model was fitted to.
    pub data: &'a [f64],
}

impl PowerLawFit<'_> {
    fn to_raw(&self, data: &Vector) -> igraph_plfit_result_t {
        igraph_plfit_result_t {
            continuous: self.continuous,
            alpha: self.alpha,
            xmin: self.xmin,
            L: self.log_likelihood,
            D: self.ks_statistic,
            data,
        }
    }

    /// Rejects models that igraph's resampling code cannot handle.
    fn check_model(&self) -> Result<()> {
        if self.data.is_empty() || self.data.iter().any(|x| !x.is_finite()) {
            return Err(Error::invalid(
                "the sample of the model must be non-empty and finite",
            ));
        }
        // plfit's discrete sampler converts heavy-tailed draws to a C `long`
        // (see `p_value`); huge samples make such overflowing draws likely.
        if !self.continuous && self.data.iter().any(|&x| x >= DISCRETE_SAMPLE_LIMIT) {
            return Err(Error::invalid(format!(
                "the sample of a discrete model must be below 2^62 = {DISCRETE_SAMPLE_LIMIT}"
            )));
        }
        if !(self.alpha > 1.0 && self.alpha.is_finite()) {
            return Err(Error::invalid(format!(
                "the exponent of the model must be finite and greater than 1, got {}",
                self.alpha
            )));
        }
        // plfit converts a discrete `xmin` to a C `long`.
        let xmin_ok = if self.continuous {
            self.xmin > 0.0 && self.xmin.is_finite()
        } else {
            (1.0..4.0e18).contains(&self.xmin)
        };
        if !xmin_ok {
            return Err(Error::invalid(format!(
                "invalid xmin for a {} power law: {}",
                if self.continuous {
                    "continuous"
                } else {
                    "discrete"
                },
                self.xmin
            )));
        }
        Ok(())
    }

    /// Computes the p-value of the fit by a (slow) resampling procedure.
    ///
    /// Many synthetic datasets are drawn: the part of the sample below `xmin`
    /// is resampled from the data itself, the part above `xmin` from the
    /// fitted power law. A power law is fitted to each of them, and the
    /// p-value is the fraction of synthetic datasets whose Kolmogorov–Smirnov
    /// statistic is *larger* than the observed one. Small p-values (e.g.
    /// below 0.1) mean that the power-law hypothesis can be rejected.
    ///
    /// The number of resampling rounds is `0.25 / precision²`: a precision of
    /// `0.01` means 2500 rounds. Results depend on the thread's default
    /// random number generator (seed it with [`rng::seed`](crate::rng::seed));
    /// if igraph was built with OpenMP, the rounds run in parallel and the
    /// results are not reproducible unless OpenMP is limited to one thread.
    ///
    /// Binds [`igraph_plfit_result_calculate_p_value`](https://igraph.org/c/html/latest/igraph-Nongraph.html#igraph_plfit_result_calculate_p_value).
    ///
    /// The fields of the struct are public, so the model is checked before
    /// calling igraph (whose resampling code crashes or loops forever on
    /// some invalid models): it must have a non-empty, finite sample, a
    /// finite `alpha > 1` (degenerate fits with `alpha = inf` are rejected),
    /// and a finite `xmin`, positive for a continuous law and in `[1, 4e18)`
    /// for a discrete one. The samples of a discrete model must also be below
    /// `2^62`.
    ///
    /// **Known upstream issue.** For discrete models, plfit draws from the
    /// fitted law with `(long) floor(pow(1 - u, -1 / (alpha - 1)) * xmin)`,
    /// relying on the conversion of too large values to a C `long` to "handle
    /// overflow" (`vendor/plfit/sampling.c`). Such a conversion is undefined
    /// behavior in C; on x86-64 it yields a negative number and the draw is
    /// retried. This cannot be ruled out from Rust without constraining
    /// `alpha`: it only happens for extremely heavy tails (`alpha` close to
    /// 1), which huge samples produce most readily, hence the `2^62` bound
    /// above.
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) if
    /// `precision` is not a positive number, is so small that the number of
    /// rounds would not fit in a 64-bit integer (below about `2.5e-10`), or
    /// so large that there would be no round at all (above `0.5`); if the
    /// model is invalid (see above); and the errors of [`power_law_fit`].
    pub fn p_value(&self, precision: f64) -> Result<f64> {
        if precision.is_nan() || precision <= 0.0 {
            return Err(Error::invalid(
                "the precision of the p-value must be positive",
            ));
        }
        // plfit converts the number of rounds to a C `long`: an out of range
        // value would be undefined behavior.
        if 0.25 / precision / precision >= 4.0e18 {
            return Err(Error::invalid(format!(
                "the precision of the p-value is too small, got {precision}"
            )));
        }
        self.check_model()?;
        let data = Vector::view(self.data);
        let raw = self.to_raw(&data);
        let mut p = 0.0;
        igraph_call!(igraph_plfit_result_calculate_p_value(
            &raw, &mut p, precision
        ))?;
        Ok(p)
    }
}

/// Fits a power-law distribution to a sample, with the maximum likelihood
/// method of Clauset, Shalizi and Newman.
///
/// `data` holds the *samples* (e.g. the degrees of the vertices of a graph),
/// not a histogram or a distribution function. For a given `xmin`, the
/// exponent `alpha` that maximizes the likelihood of the samples `>= xmin`
/// is returned. The threshold is:
///
/// - `None`: estimated, choosing the `xmin` for which the
///   Kolmogorov–Smirnov distance between the fitted law and the sample is
///   the smallest (slower, as every distinct value is tried);
/// - `Some(x)`: fixed to `x`; samples below `x` are ignored. `x` must be
///   positive for a continuous law and at least 1 for a discrete one (so
///   `Some(1.0)` uses all the positive integer samples); `Some(0.0)` is
///   rejected by igraph.
///
/// A discrete power law is fitted if all samples are integers, unless
/// `force_continuous` is `true`; a continuous one otherwise.
///
/// Degenerate samples give degenerate or failed fits. For a discrete law
/// with no sample reaching the given `xmin`, igraph returns `alpha = inf`
/// and a NaN log-likelihood (check `alpha.is_finite()` when `xmin` is
/// chosen by hand); the continuous fit fails with
/// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) instead, and
/// a single sample makes both fail.
///
/// See also [`Graph::degree`](crate::Graph::degree) (the usual sample to
/// fit), and the generators of scale-free graphs
/// [`Graph::barabasi_game`](crate::Graph::barabasi_game) and
/// [`Graph::static_power_law_game`](crate::Graph::static_power_law_game).
///
/// Reference: A. Clauset, C. R. Shalizi and M. E. J. Newman, *Power-law
/// distributions in empirical data*, SIAM Review 51(4):661–703, 2009.
///
/// Binds [`igraph_power_law_fit`](https://igraph.org/c/html/latest/igraph-Nongraph.html#igraph_power_law_fit).
/// Time complexity: `O(n log n)` in the continuous case with a fixed `xmin`;
/// the discrete case is dominated by an L-BFGS optimization; estimating
/// `xmin` multiplies the cost by the number of distinct samples.
///
/// # Errors
/// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) for invalid
/// data (e.g. no samples, NaN or infinite samples, an `xmin` below 1 for
/// discrete or not positive for continuous samples, discrete samples or
/// `xmin` of `2^62` or more), and other kinds for numerical failures of the
/// fitting procedure.
///
/// Integer samples of `2^62` (about `4.6e18`) or more are only accepted with
/// `force_continuous = true`: for huge discrete samples plfit's L-BFGS
/// optimization fails, and igraph 1.0.1 then reads the error message from a
/// stack buffer that no longer exists. The bound matches the one of
/// [`PowerLawFit::p_value`], so every discrete fit can be tested.
///
/// # Examples
///
/// Continuous samples drawn by inverse transform sampling from a power law
/// with `alpha = 2.5` and `xmin = 1`:
///
/// ```
/// use igraph::{misc, prelude::*};
///
/// rng::seed(7)?;
/// let sample: Vec<f64> =
///     (0..5000).map(|_| (1.0 - rng::uniform01()).powf(-1.0 / 1.5)).collect();
/// let fit = misc::power_law_fit(&sample, Some(1.0), false)?;
/// assert!(fit.continuous);
/// assert!((fit.alpha - 2.5).abs() < 0.1);
/// assert_eq!(fit.xmin, 1.0);
/// # Ok::<(), igraph::Error>(())
/// ```
///
/// The degrees of a Barabási–Albert graph (igraph's
/// `examples/simple/igraph_power_law_fit.c`, with the same seed and output):
///
/// ```
/// use igraph::{games::BarabasiOptions, misc, prelude::*};
///
/// rng::seed(42)?;
/// let options = BarabasiOptions {
///     m: 2,
///     algorithm: BarabasiAlgorithm::Bag,
///     ..Default::default()
/// };
/// let g = Graph::barabasi_game(10_000, &options)?;
/// let degrees: Vec<f64> = g
///     .degree(.., NeighborMode::All, Loops::None)?
///     .into_iter()
///     .map(|d| d as f64)
///     .collect();
/// let fit = misc::power_law_fit(&degrees, None, false)?;
/// assert!(!fit.continuous); // integer samples: a discrete law
/// assert_eq!(fit.xmin, 7.0);
/// assert!((fit.alpha - 3.04393).abs() < 1e-5);
/// # Ok::<(), igraph::Error>(())
/// ```
pub fn power_law_fit(
    data: &[f64],
    xmin: Option<f64>,
    force_continuous: bool,
) -> Result<PowerLawFit<'_>> {
    let xmin = match xmin {
        None => -1.0,
        Some(x) if x >= 0.0 => x,
        Some(x) => {
            return Err(Error::invalid(format!(
                "xmin must be non-negative, got {x}"
            )));
        }
    };
    // When the L-BFGS optimization of the discrete fit fails, plfit formats
    // its error message into a stack buffer that igraph reads after it is
    // gone (a use-after-return in igraph 1.0.1). It fails for infinite
    // samples, and for huge ones (seen from about 1e158 on): reject both,
    // with a wide margin for the latter.
    if data.iter().any(|x| !x.is_finite()) {
        return Err(Error::invalid(
            "the data must not contain NaN or infinite values",
        ));
    }
    // igraph fits a discrete law iff all samples are integers.
    let discrete = !force_continuous && data.iter().all(|x| x.trunc() == *x);
    if discrete
        && (xmin >= DISCRETE_SAMPLE_LIMIT || data.iter().any(|&x| x >= DISCRETE_SAMPLE_LIMIT))
    {
        return Err(Error::invalid(format!(
            "discrete samples and xmin must be below 2^62 = {DISCRETE_SAMPLE_LIMIT} \
             (use force_continuous for larger values)"
        )));
    }
    let view = Vector::view(data);
    let mut raw = igraph_plfit_result_t {
        continuous: false,
        alpha: 0.0,
        xmin: 0.0,
        L: 0.0,
        D: 0.0,
        data: std::ptr::null(),
    };
    igraph_call!(igraph_power_law_fit(
        view.as_ptr(),
        &mut raw,
        xmin,
        force_continuous
    ))?;
    Ok(PowerLawFit {
        continuous: raw.continuous,
        alpha: raw.alpha,
        xmin: raw.xmin,
        log_likelihood: raw.L,
        ks_statistic: raw.D,
        data,
    })
}

/// Running (moving) mean of `data` over windows of `binwidth` consecutive
/// values.
///
/// The result has `data.len() - binwidth + 1` elements: element `i` is the
/// mean of `data[i..i + binwidth]`.
///
/// Binds [`igraph_running_mean`](https://igraph.org/c/html/latest/igraph-Nongraph.html#igraph_running_mean).
/// Time complexity: `O(n)`.
///
/// # Errors
/// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) if
/// `binwidth` is zero or larger than `data.len()`.
///
/// # Examples
///
/// ```
/// use igraph::misc;
/// assert_eq!(misc::running_mean(&[1.0, 2.0, 3.0, 4.0, 5.0], 2)?, vec![1.5, 2.5, 3.5, 4.5]);
/// # Ok::<(), igraph::Error>(())
/// ```
pub fn running_mean(data: &[f64], binwidth: usize) -> Result<Vec<f64>> {
    let view = Vector::view(data);
    let mut res = Vector::new();
    igraph_call!(igraph_running_mean(
        view.as_ptr(),
        &mut res,
        binwidth as igraph_int_t
    ))?;
    Ok(res.into())
}

/// Draws `length` distinct integers uniformly at random from the closed
/// interval `[low, high]`, returned in increasing order.
///
/// It uses Vitter's sequential sampling algorithm ("Method D"), which runs in
/// expected `O(length)` time and memory, regardless of the size of the
/// interval: ideal to select a few edges out of a huge set of candidates.
/// It draws from the thread's default random number generator. An empty
/// sample (`length == 0`) is always empty, also when `low == high` (igraph
/// 1.0.0 and 1.0.1 return `[low]` there; the wrapper corrects this).
///
/// See also [`rng::shuffle`](crate::rng::shuffle) for random permutations,
/// and [`Graph::erdos_renyi_game_gnm`](crate::Graph::erdos_renyi_game_gnm),
/// which picks its edges among all vertex pairs with the same method.
///
/// Reference: J. S. Vitter, *An efficient algorithm for sequential random
/// sampling*, ACM Transactions on Mathematical Software 13(1):58–67, 1987.
///
/// Binds [`igraph_random_sample`](https://igraph.org/c/html/latest/igraph-Nongraph.html#igraph_random_sample).
///
/// # Errors
/// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) if
/// `low > high` or `length` exceeds the size of the interval;
/// [`ErrorKind::Overflow`](crate::ErrorKind::Overflow) for a non-empty
/// sample from an interval with more than `i64::MAX` elements.
///
/// # Examples
///
/// ```
/// use igraph::{misc, prelude::*};
/// rng::seed(1)?;
/// let s = misc::random_sample(0, 1_000_000_000_000, 5)?;
/// assert_eq!(s.len(), 5);
/// assert!(s.windows(2).all(|w| w[0] < w[1]));
/// # Ok::<(), igraph::Error>(())
/// ```
pub fn random_sample(low: i64, high: i64, length: usize) -> Result<Vec<i64>> {
    if low > high {
        return Err(Error::invalid(format!(
            "the lower limit {low} is greater than the upper limit {high}"
        )));
    }
    let Ok(length) = igraph_int_t::try_from(length) else {
        return Err(Error::invalid("sample size exceeds size of candidate pool"));
    };
    // igraph 1.0.0 and 1.0.1 return `[low]` when `low == high`, even for
    // `length == 0`.
    if length == 0 {
        return Ok(Vec::new());
    }
    // igraph computes `high + (-low)`, which overflows (undefined behavior
    // in C) for `low == i64::MIN`: sample from `[0, high - low]` and shift.
    // The algorithm only depends on the size of the interval, so the result
    // (and the use of the random number generator) is the same.
    let Some(span) = high.checked_sub(low) else {
        return Err(Error::new(
            ErrorKind::Overflow,
            "the interval has more than i64::MAX elements",
        ));
    };
    let mut res = VectorInt::new();
    igraph_call!(igraph_random_sample(&mut res, 0, span, length))?;
    Ok(res.iter().map(|&x| x + low).collect())
}

/// Whether `a` and `b` are equal up to the relative tolerance `eps`, i.e.
/// whether `|a - b| / (|a| + |b|) < eps` (with sensible handling of zeros,
/// infinities and NaNs, see [`cmp_epsilon`]).
///
/// Binds [`igraph_almost_equals`](https://igraph.org/c/html/latest/igraph-Nongraph.html#igraph_almost_equals).
///
/// # Examples
///
/// ```
/// use igraph::misc::almost_equals;
/// assert!(almost_equals(0.1 + 0.2, 0.3, 1e-12));
/// assert!(!almost_equals(1.0, 1.1, 1e-3));
/// ```
pub fn almost_equals(a: f64, b: f64, eps: f64) -> bool {
    unsafe { igraph_almost_equals(a, b, eps) }
}

/// Three-way comparison of `a` and `b` with the relative tolerance `eps`:
/// [`Ordering::Equal`] when `|a - b| / (|a| + |b|) < eps`, otherwise the
/// natural order of the two numbers.
///
/// Infinities are equal to infinities of the same sign and are larger (or
/// smaller) than every finite number, whatever the tolerance. NaN is never
/// equal to anything (not even NaN), but the direction of the resulting
/// ordering is unspecified. An `eps` of zero means exact comparison;
/// negative values of `eps` give unspecified results.
///
/// Binds [`igraph_cmp_epsilon`](https://igraph.org/c/html/latest/igraph-Nongraph.html#igraph_cmp_epsilon).
///
/// # Examples
///
/// ```
/// use igraph::misc::cmp_epsilon;
/// use std::cmp::Ordering;
///
/// assert_eq!(cmp_epsilon(1.0, 1.0 + 1e-12, 1e-9), Ordering::Equal);
/// assert_eq!(cmp_epsilon(1.0, 2.0, 1e-9), Ordering::Less);
/// assert_eq!(cmp_epsilon(f64::INFINITY, 1e300, 0.5), Ordering::Greater);
/// ```
pub fn cmp_epsilon(a: f64, b: f64, eps: f64) -> Ordering {
    unsafe { igraph_cmp_epsilon(a, b, eps) }.cmp(&0)
}

/// The version of the igraph C library this crate is linked against, as
/// returned by [`version`].
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Version {
    /// The major version, e.g. `1` for `"1.0.1"`.
    pub major: i32,
    /// The minor version, e.g. `0` for `"1.0.1"`.
    pub minor: i32,
    /// The patch (subminor) version, e.g. `1` for `"1.0.1"`.
    pub patch: i32,
    /// The full version string: three dot-separated numbers, possibly
    /// followed by a dash-separated pre-release suffix
    /// (e.g. `"0.10.13-14-g997f59ad7"`).
    pub string: String,
}

impl Version {
    /// The `(major, minor, patch)` triple, handy for comparisons such as
    /// `version().triple() >= (1, 0, 0)`.
    pub fn triple(&self) -> (i32, i32, i32) {
        (self.major, self.minor, self.patch)
    }
}

impl fmt::Display for Version {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.string)
    }
}

/// The version of the igraph C library in use (at run time).
///
/// This is the library actually loaded by the dynamic linker, which may
/// differ from the one whose headers the crate was compiled against (e.g.
/// when `LD_LIBRARY_PATH` points to another build): this crate targets
/// igraph 1.0.1.
///
/// Binds [`igraph_version`](https://igraph.org/c/html/latest/igraph-Nongraph.html#igraph_version).
///
/// # Examples
///
/// ```
/// let v = igraph::misc::version();
/// assert!(v.triple() >= (1, 0, 0));
/// assert!(v.to_string().starts_with(&format!("{}.{}.{}", v.major, v.minor, v.patch)));
/// ```
pub fn version() -> Version {
    let mut string: *const std::ffi::c_char = std::ptr::null();
    let (mut major, mut minor, mut patch) = (0, 0, 0);
    unsafe { igraph_version(&mut string, &mut major, &mut minor, &mut patch) };
    Version {
        major,
        minor,
        patch,
        string: lossy(string),
    }
}

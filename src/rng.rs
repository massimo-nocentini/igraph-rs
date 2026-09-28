//! Random number generation (`igraph_random.h`).
//!
//! Every randomized igraph function (the random graph generators of
//! [`games`](crate::games), randomized community detection, layouts with a
//! random start, [`Vector::shuffle`](crate::vector::Vector::shuffle), ...)
//! draws from the calling thread's *default* random number generator. Each
//! thread gets its own (a PCG32 generator, igraph's default algorithm,
//! seeded randomly the first time the thread calls into igraph), so threads
//! never share random state, and [`seed`] makes the results of a thread
//! reproducible, whatever the other threads do.
//!
//! The generators are igraph's own, so a seed gives the very numbers that
//! igraph's C test suite pins (`tests/unit/igraph_rng_get_integer.out`):
//!
//! ```
//! use igraph::prelude::*;
//!
//! rng::seed(42).unwrap();
//! let a: Vec<i64> = (0..10).map(|_| rng::integer(10, 100)).collect();
//! assert_eq!(a, vec![22, 59, 86, 99, 99, 77, 93, 12, 20, 62]);
//! rng::seed(42).unwrap();
//! let b: Vec<i64> = (0..10).map(|_| rng::integer(10, 100)).collect();
//! assert_eq!(a, b);
//!
//! // Random graphs are reproducible too.
//! rng::seed(7).unwrap();
//! let g = Graph::erdos_renyi_game_gnp(30, 0.2, false, EdgeTypeSw::Simple, false).unwrap();
//! rng::seed(7).unwrap();
//! let h = Graph::erdos_renyi_game_gnp(30, 0.2, false, EdgeTypeSw::Simple, false).unwrap();
//! assert_eq!(g, h);
//! ```
//!
//! Independent generators of a chosen [`RngType`] are created as owned
//! [`Rng`] values, and can be installed as the default for the duration of a
//! closure with [`Rng::scoped`].
//!
//! | C function (`igraph_random.h`) | Default generator | Owned [`Rng`] |
//! |------------|-------------------|---------------|
//! | `igraph_rng_init` / `_destroy` | (per thread, automatic) | [`Rng::new`] / [`Drop`] |
//! | `igraph_rng_seed` | [`seed`] | [`Rng::set_seed`] |
//! | `igraph_rng_name`, `_bits`, `_max` | [`name`], [`bits`], [`max`] | [`Rng::name`], [`Rng::bits`], [`Rng::max`] |
//! | (the `is_seeded` field) | | [`Rng::is_seeded`] |
//! | `igraph_rng_get_integer` | [`integer`] | [`Rng::get_integer`] |
//! | `igraph_rng_get_unif(01)` | [`uniform`], [`uniform01`] | [`Rng::get_unif`], [`Rng::get_unif01`] |
//! | `igraph_rng_get_bool` | [`boolean`] | [`Rng::get_bool`] |
//! | `igraph_rng_get_normal` | [`normal`] | [`Rng::get_normal`] |
//! | `igraph_rng_get_geom`, `_binom`, `_exp`, `_gamma`, `_pois` | [`geometric`], [`binomial`], [`exponential`], [`gamma`], [`poisson`] | [`Rng::get_geom`], [`Rng::get_binom`], [`Rng::get_exp`], [`Rng::get_gamma`], [`Rng::get_pois`] |
//! | `igraph_rng_default`, `igraph_rng_set_default` | | [`Rng::scoped`] |
//! | (Fisher-Yates in Rust) | [`shuffle`] | |
//!
//! See also [`games`](crate::games) for random graph models, and
//! [`misc::sample_sphere_surface`](crate::misc::sample_sphere_surface),
//! [`misc::sample_sphere_volume`](crate::misc::sample_sphere_volume) and
//! [`misc::sample_dirichlet`](crate::misc::sample_dirichlet) (also
//! available as methods of an owned [`Rng`]) for random points and
//! [`misc::random_sample`](crate::misc::random_sample) for sampling integers
//! without replacement.

use crate::{error::Result, ffi::*, igraph_call};
use std::mem::MaybeUninit;

/// An owned random number generator (`igraph_rng_t`).
pub type Rng = igraph_rng_t;

/// The random number generator algorithms shipped with igraph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RngType {
    /// Mersenne Twister MT19937.
    Mt19937,
    /// The generator of the GNU C library (`random()`).
    Glibc2,
    /// PCG32 (the algorithm of igraph's default generator).
    Pcg32,
    /// PCG64.
    Pcg64,
}

impl RngType {
    /// All the generator types, in declaration order.
    pub const ALL: [RngType; 4] = [Self::Mt19937, Self::Glibc2, Self::Pcg32, Self::Pcg64];

    fn raw(self) -> *const igraph_rng_type_t {
        match self {
            Self::Mt19937 => &raw const igraph_rngtype_mt19937,
            Self::Glibc2 => &raw const igraph_rngtype_glibc2,
            Self::Pcg32 => &raw const igraph_rngtype_pcg32,
            Self::Pcg64 => &raw const igraph_rngtype_pcg64,
        }
    }
}

/// Installs a fresh, randomly seeded PCG32 generator (igraph's default
/// algorithm) as the default generator of the calling thread. Called once per
/// thread by [`ensure_init`](crate::error::ensure_init).
///
/// The generator is intentionally leaked: igraph keeps a raw pointer to it
/// in thread-local storage that may outlive any Rust thread-local destructor.
pub(crate) fn install_thread_default_rng() {
    use std::hash::{BuildHasher, Hasher};
    let mut raw = MaybeUninit::<igraph_rng_t>::zeroed();
    let code = unsafe { igraph_rng_init(raw.as_mut_ptr(), RngType::Pcg32.raw()) };
    if code != igraph_error_type_t_IGRAPH_SUCCESS {
        // Out of memory while setting up a thread: keep igraph's default.
        return;
    }
    let rng: &'static mut igraph_rng_t = Box::leak(Box::new(unsafe { raw.assume_init() }));
    // `RandomState` is seeded from the OS once per thread and varies per call.
    let mut h = std::collections::hash_map::RandomState::new().build_hasher();
    h.write_u64(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos() as u64),
    );
    unsafe {
        igraph_rng_seed(rng, h.finish());
        igraph_rng_set_default(rng);
    }
}

fn default_rng() -> *mut igraph_rng_t {
    crate::error::ensure_init();
    unsafe { igraph_rng_default() }
}

/// Seeds the default random number generator of the calling thread
/// ([`igraph_rng_seed`](https://igraph.org/c/html/latest/igraph-Random.html#igraph_rng_seed)).
///
/// Only the calling thread is affected: every thread has its own default
/// generator, so seeded computations are reproducible even when other
/// threads use random numbers at the same time. A generator installed with
/// [`Rng::scoped`] is the default while it is in scope, so it is the one
/// seeded then.
///
/// # Errors
/// Only if igraph fails to seed the generator (it never does for the
/// generators shipped with igraph).
pub fn seed(seed: u64) -> Result<()> {
    igraph_call!(igraph_rng_seed(default_rng(), seed))
}

/// igraph 1.0.0 and 1.0.1 only `assert` the order of the bounds (aborting,
/// or computing garbage in release builds of the C library): check them in
/// Rust.
#[track_caller]
fn check_int_bounds(min: i64, max: i64) {
    assert!(min <= max, "empty integer interval [{min}, {max}]");
}

/// The bounds of a uniform real must be finite (an infinite width makes
/// igraph 1.0.0 and 1.0.1 loop, practically forever) and ordered (only
/// `assert`ed by igraph).
#[track_caller]
fn check_real_bounds(min: f64, max: f64) {
    assert!(
        min.is_finite() && max.is_finite() && min <= max && (max - min).is_finite(),
        "invalid real interval [{min}, {max})"
    );
}

/// A uniform random integer in the closed interval `[min, max]`
/// ([`igraph_rng_get_integer`](https://igraph.org/c/html/latest/igraph-Random.html#igraph_rng_get_integer)).
///
/// # Panics
/// If `min > max`.
#[track_caller]
pub fn integer(min: i64, max: i64) -> i64 {
    check_int_bounds(min, max);
    unsafe { igraph_rng_get_integer(default_rng(), min, max) }
}

/// A uniform random real in `[min, max)` (`min` itself if `min == max`)
/// ([`igraph_rng_get_unif`](https://igraph.org/c/html/latest/igraph-Random.html#igraph_rng_get_unif)).
///
/// # Panics
/// If a bound is not finite, `max - min` overflows, or `min > max`.
#[track_caller]
pub fn uniform(min: f64, max: f64) -> f64 {
    check_real_bounds(min, max);
    unsafe { igraph_rng_get_unif(default_rng(), min, max) }
}

/// A uniform random real in the half-open interval `[0, 1)`
/// ([`igraph_rng_get_unif01`](https://igraph.org/c/html/latest/igraph-Random.html#igraph_rng_get_unif01)).
pub fn uniform01() -> f64 {
    unsafe { igraph_rng_get_unif01(default_rng()) }
}

/// A random boolean, `true` with probability 1/2
/// ([`igraph_rng_get_bool`](https://igraph.org/c/html/latest/igraph-Random.html#igraph_rng_get_bool)).
pub fn boolean() -> bool {
    unsafe { igraph_rng_get_bool(default_rng()) }
}

/// A normally distributed random real with mean `mean` and standard
/// deviation `sd`, i.e. with density proportional to
/// `exp(-(x - mean)² / (2 sd²))`
/// ([`igraph_rng_get_normal`](https://igraph.org/c/html/latest/igraph-Random.html#igraph_rng_get_normal)).
pub fn normal(mean: f64, sd: f64) -> f64 {
    unsafe { igraph_rng_get_normal(default_rng(), mean, sd) }
}

/// A geometrically distributed random number (failures before the first
/// success) with success probability `p` in `(0, 1]`; NaN for an invalid `p`
/// ([`igraph_rng_get_geom`](https://igraph.org/c/html/latest/igraph-Random.html#igraph_rng_get_geom)).
pub fn geometric(p: f64) -> f64 {
    unsafe { igraph_rng_get_geom(default_rng(), p) }
}

/// A binomially distributed random number: the number of successes in `n`
/// independent trials of success probability `p`, i.e. `k` with
/// probability `C(n, k) p^k (1 - p)^(n - k)`
/// ([`igraph_rng_get_binom`](https://igraph.org/c/html/latest/igraph-Random.html#igraph_rng_get_binom)).
pub fn binomial(n: i64, p: f64) -> f64 {
    unsafe { igraph_rng_get_binom(default_rng(), n, p) }
}

/// An exponentially distributed random number with rate `rate` (mean
/// `1 / rate`); NaN for a non-positive rate, `0` for an infinite one
/// ([`igraph_rng_get_exp`](https://igraph.org/c/html/latest/igraph-Random.html#igraph_rng_get_exp)).
pub fn exponential(rate: f64) -> f64 {
    unsafe { igraph_rng_get_exp(default_rng(), rate) }
}

/// A gamma distributed random number with density proportional to
/// `x^(shape - 1) exp(-x / scale)` (mean `shape * scale`)
/// ([`igraph_rng_get_gamma`](https://igraph.org/c/html/latest/igraph-Random.html#igraph_rng_get_gamma)).
pub fn gamma(shape: f64, scale: f64) -> f64 {
    unsafe { igraph_rng_get_gamma(default_rng(), shape, scale) }
}

/// A Poisson distributed random number with mean `mean`; NaN if `mean` is
/// negative or NaN
/// ([`igraph_rng_get_pois`](https://igraph.org/c/html/latest/igraph-Random.html#igraph_rng_get_pois)).
pub fn poisson(mean: f64) -> f64 {
    unsafe { igraph_rng_get_pois(default_rng(), mean) }
}

/// The name of the algorithm of the thread's default generator
/// ([`igraph_rng_name`](https://igraph.org/c/html/latest/igraph-Random.html#igraph_rng_name)),
/// e.g. `"PCG64"`.
pub fn name() -> String {
    let ptr = unsafe { igraph_rng_name(default_rng()) };
    unsafe { std::ffi::CStr::from_ptr(ptr) }
        .to_string_lossy()
        .into_owned()
}

/// Number of random bits produced at once by the default generator
/// ([`igraph_rng_bits`](https://igraph.org/c/html/latest/igraph-Random.html#igraph_rng_bits)).
pub fn bits() -> u32 {
    unsafe { igraph_rng_bits(default_rng()) as u32 }
}

/// The largest integer the default generator can produce natively
/// ([`igraph_rng_max`](https://igraph.org/c/html/latest/igraph-Random.html#igraph_rng_max)).
pub fn max() -> u64 {
    unsafe { igraph_rng_max(default_rng()) }
}

/// Shuffles a slice in place (Fisher-Yates) with the default generator, so
/// that the outcome is reproducible with [`seed`]; the slice counterpart of
/// [`Vector::shuffle`](crate::vector::Vector::shuffle) (`igraph_vector_shuffle`),
/// which runs the same algorithm: for the same seed both produce the same
/// permutation.
///
/// ```
/// use igraph::prelude::*;
/// let mut a: Vec<u32> = (0..10).collect();
/// rng::seed(1).unwrap();
/// rng::shuffle(&mut a);
/// let mut b: Vec<u32> = (0..10).collect();
/// rng::seed(1).unwrap();
/// rng::shuffle(&mut b);
/// assert_eq!(a, b);
/// a.sort();
/// assert_eq!(a, (0..10).collect::<Vec<_>>());
/// ```
pub fn shuffle<T>(slice: &mut [T]) {
    for i in (1..slice.len()).rev() {
        let j = integer(0, i as i64) as usize;
        slice.swap(i, j);
    }
}

impl igraph_rng_t {
    /// Creates a new generator of the given type, seeded with `seed`
    /// ([`igraph_rng_init`](https://igraph.org/c/html/latest/igraph-Random.html#igraph_rng_init) and
    /// [`igraph_rng_seed`](https://igraph.org/c/html/latest/igraph-Random.html#igraph_rng_seed)).
    ///
    /// Two generators of the same type and seed produce the same numbers,
    /// on any thread:
    ///
    /// ```
    /// use igraph::prelude::*;
    /// // Pinned by igraph's own tests (`rng_init_destroy_max_bits_name_set_default.out`).
    /// let mut mt = Rng::new(RngType::Mt19937, 42).unwrap();
    /// let draws: Vec<i64> = (0..5).map(|_| mt.get_integer(0, 100)).collect();
    /// assert_eq!(draws, vec![37, 80, 96, 18, 73]);
    /// ```
    ///
    /// # Errors
    /// [`ErrorKind::OutOfMemory`](crate::error::ErrorKind::OutOfMemory) if
    /// the state cannot be allocated.
    pub fn new(kind: RngType, seed: u64) -> Result<Self> {
        crate::error::ensure_init();
        let mut raw = MaybeUninit::<Self>::zeroed();
        igraph_call!(igraph_rng_init(raw.as_mut_ptr(), kind.raw()))?;
        let mut rng = unsafe { raw.assume_init() };
        rng.set_seed(seed)?;
        Ok(rng)
    }

    /// Re-seeds this generator
    /// ([`igraph_rng_seed`](https://igraph.org/c/html/latest/igraph-Random.html#igraph_rng_seed)): the numbers it produces
    /// afterwards depend only on its type and on `seed`.
    pub fn set_seed(&mut self, seed: u64) -> Result<()> {
        igraph_call!(igraph_rng_seed(self, seed))
    }

    /// The name of the algorithm, e.g. `"MT19937"`, `"LIBC"`, `"PCG32"` or
    /// `"PCG64"` ([`igraph_rng_name`](https://igraph.org/c/html/latest/igraph-Random.html#igraph_rng_name)).
    pub fn name(&self) -> String {
        let ptr = unsafe { igraph_rng_name(self) };
        unsafe { std::ffi::CStr::from_ptr(ptr) }
            .to_string_lossy()
            .into_owned()
    }

    /// A uniform random integer in the closed interval `[min, max]`
    /// ([`igraph_rng_get_integer`](https://igraph.org/c/html/latest/igraph-Random.html#igraph_rng_get_integer)).
    ///
    /// # Panics
    /// If `min > max`.
    #[track_caller]
    pub fn get_integer(&mut self, min: i64, max: i64) -> i64 {
        check_int_bounds(min, max);
        unsafe { igraph_rng_get_integer(self, min, max) }
    }

    /// A uniform random real in `[min, max)`
    /// ([`igraph_rng_get_unif`](https://igraph.org/c/html/latest/igraph-Random.html#igraph_rng_get_unif)).
    ///
    /// # Panics
    /// If a bound is not finite, `max - min` overflows, or `min > max`.
    #[track_caller]
    pub fn get_unif(&mut self, min: f64, max: f64) -> f64 {
        check_real_bounds(min, max);
        unsafe { igraph_rng_get_unif(self, min, max) }
    }

    /// A uniform random real in `[0, 1)`
    /// ([`igraph_rng_get_unif01`](https://igraph.org/c/html/latest/igraph-Random.html#igraph_rng_get_unif01)).
    pub fn get_unif01(&mut self) -> f64 {
        unsafe { igraph_rng_get_unif01(self) }
    }

    /// A normally distributed random real with mean `mean` and standard
    /// deviation `sd`
    /// ([`igraph_rng_get_normal`](https://igraph.org/c/html/latest/igraph-Random.html#igraph_rng_get_normal)).
    pub fn get_normal(&mut self, mean: f64, sd: f64) -> f64 {
        unsafe { igraph_rng_get_normal(self, mean, sd) }
    }

    /// A random boolean ([`igraph_rng_get_bool`](https://igraph.org/c/html/latest/igraph-Random.html#igraph_rng_get_bool)).
    pub fn get_bool(&mut self) -> bool {
        unsafe { igraph_rng_get_bool(self) }
    }

    /// A geometric random number: the number of failures before the first
    /// success, with success probability `p`
    /// ([`igraph_rng_get_geom`](https://igraph.org/c/html/latest/igraph-Random.html#igraph_rng_get_geom)).
    pub fn get_geom(&mut self, p: f64) -> f64 {
        unsafe { igraph_rng_get_geom(self, p) }
    }

    /// A binomial random number with `n` trials of probability `p`
    /// ([`igraph_rng_get_binom`](https://igraph.org/c/html/latest/igraph-Random.html#igraph_rng_get_binom)).
    pub fn get_binom(&mut self, n: i64, p: f64) -> f64 {
        unsafe { igraph_rng_get_binom(self, n, p) }
    }

    /// An exponential random number with the given `rate`
    /// ([`igraph_rng_get_exp`](https://igraph.org/c/html/latest/igraph-Random.html#igraph_rng_get_exp)).
    pub fn get_exp(&mut self, rate: f64) -> f64 {
        unsafe { igraph_rng_get_exp(self, rate) }
    }

    /// A gamma random number with the given `shape` and `scale`
    /// ([`igraph_rng_get_gamma`](https://igraph.org/c/html/latest/igraph-Random.html#igraph_rng_get_gamma)).
    pub fn get_gamma(&mut self, shape: f64, scale: f64) -> f64 {
        unsafe { igraph_rng_get_gamma(self, shape, scale) }
    }

    /// A Poisson random number with the given `mean`
    /// ([`igraph_rng_get_pois`](https://igraph.org/c/html/latest/igraph-Random.html#igraph_rng_get_pois)).
    pub fn get_pois(&mut self, mean: f64) -> f64 {
        unsafe { igraph_rng_get_pois(self, mean) }
    }

    /// Number of random bits produced at once by this generator
    /// ([`igraph_rng_bits`](https://igraph.org/c/html/latest/igraph-Random.html#igraph_rng_bits)).
    pub fn bits(&self) -> u32 {
        unsafe { igraph_rng_bits(self) as u32 }
    }

    /// The largest integer this generator produces natively
    /// ([`igraph_rng_max`](https://igraph.org/c/html/latest/igraph-Random.html#igraph_rng_max)).
    pub fn max(&self) -> u64 {
        unsafe { igraph_rng_max(self) }
    }

    /// Whether the generator has been seeded (always true for generators
    /// created with [`Rng::new`]): the `is_seeded` flag that
    /// `igraph_rng_seed` sets, and that `igraph_setup` checks before seeding
    /// the default generator.
    pub fn is_seeded(&self) -> bool {
        self.is_seeded
    }

    /// Runs `f` with this generator installed as the thread's default one
    /// ([`igraph_rng_set_default`](https://igraph.org/c/html/latest/igraph-Random.html#igraph_rng_set_default)), restoring the
    /// previous default afterwards, even if `f` panics.
    ///
    /// Every randomized igraph function called by `f` on this thread (and
    /// the free functions of this module) then draws from `self`; other
    /// threads are not affected.
    ///
    /// ```
    /// use igraph::prelude::*;
    /// let mut rng = Rng::new(RngType::Mt19937, 7).unwrap();
    /// let x = rng.scoped(|| rng::integer(0, 1_000_000));
    /// let mut again = Rng::new(RngType::Mt19937, 7).unwrap();
    /// assert_eq!(again.scoped(|| rng::integer(0, 1_000_000)), x);
    /// ```
    pub fn scoped<R>(&mut self, f: impl FnOnce() -> R) -> R {
        struct Restore(*mut igraph_rng_t);
        impl Drop for Restore {
            fn drop(&mut self) {
                unsafe { igraph_rng_set_default(self.0) };
            }
        }
        crate::error::ensure_init();
        let _restore = Restore(unsafe { igraph_rng_set_default(self) });
        f()
    }
}

impl Drop for igraph_rng_t {
    fn drop(&mut self) {
        if !self.state.is_null() {
            unsafe { igraph_rng_destroy(self) };
        }
    }
}

unsafe impl Send for igraph_rng_t {}

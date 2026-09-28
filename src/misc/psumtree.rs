//! Partial prefix-sum trees (`igraph_psumtree.h`).

use crate::{
    error::{Error, Result},
    ffi::*,
    igraph_call,
};
use std::mem::MaybeUninit;

/// A partial prefix-sum tree (`igraph_psumtree_t`): a fixed number of items,
/// each with a non-negative weight, supporting weight updates and
/// *weighted sampling* in `O(log n)`.
///
/// It is the data structure igraph uses internally to draw vertices with
/// probability proportional to changing weights (e.g. in preferential
/// attachment models or in the [SIR simulation](crate::Graph::sir)).
/// [`search`](PsumTree::search) maps a number in `[0, sum)` to the item
/// whose cumulative weight interval contains it, so feeding it uniform
/// random numbers samples items proportionally to their weights; this is
/// what [`sample`](PsumTree::sample) does.
///
/// For one-off sampling from a fixed distribution, the functions of
/// [`crate::rng`] are simpler; the tree pays off when weights change
/// between draws.
///
/// Binds the `igraph_psumtree_*` functions, see the
/// [igraph manual](https://igraph.org/c/html/latest/igraph-Data-structures.html#igraph_psumtree_init).
///
/// # Examples
///
/// ```
/// use igraph::misc::PsumTree;
///
/// let mut tree = PsumTree::from_weights(&[1.0, 0.0, 3.0])?;
/// assert_eq!(tree.sum(), 4.0);
/// // [0, 1) -> item 0; [1, 4) -> item 2 (item 1 has no weight)
/// assert_eq!(tree.search(0.5)?, 0);
/// assert_eq!(tree.search(1.0)?, 2);
/// tree.update(1, 10.0)?;
/// assert_eq!(tree.search(1.0)?, 1);
/// assert_eq!(tree.get(1), Some(10.0));
/// # Ok::<(), igraph::Error>(())
/// ```
pub type PsumTree = igraph_psumtree_t;

impl igraph_psumtree_t {
    /// Creates a tree with `size` items, all of weight zero
    /// (`igraph_psumtree_init`).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) if `size`
    /// is zero.
    pub fn new(size: usize) -> Result<Self> {
        if size == 0 {
            return Err(Error::invalid(
                "a prefix-sum tree must have at least one item",
            ));
        }
        let size = igraph_int_t::try_from(size)
            .map_err(|_| Error::invalid("too many items for a prefix-sum tree"))?;
        crate::error::ensure_init();
        let mut raw = MaybeUninit::<Self>::zeroed();
        igraph_call!(igraph_psumtree_init(raw.as_mut_ptr(), size))?;
        Ok(unsafe { raw.assume_init() })
    }

    /// Creates a tree whose items have the given weights.
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) if
    /// `weights` is empty, or a weight is negative or not finite.
    pub fn from_weights(weights: &[f64]) -> Result<Self> {
        let mut tree = Self::new(weights.len())?;
        for (i, &w) in weights.iter().enumerate() {
            tree.update(i, w)?;
        }
        Ok(tree)
    }

    /// The number of items (`igraph_psumtree_size`).
    pub fn len(&self) -> usize {
        unsafe { igraph_psumtree_size(self) as usize }
    }

    /// Whether the tree has no items (never true for a successfully created tree).
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The weight of item `index`, or `None` if out of bounds
    /// (`igraph_psumtree_get`). `O(1)`.
    pub fn get(&self, index: usize) -> Option<f64> {
        (index < self.len()).then(|| unsafe { igraph_psumtree_get(self, index as igraph_int_t) })
    }

    /// The weights of all the items, in order.
    pub fn weights(&self) -> Vec<f64> {
        (0..self.len())
            .map(|i| unsafe { igraph_psumtree_get(self, i as igraph_int_t) })
            .collect()
    }

    /// The total weight of the items (`igraph_psumtree_sum`). `O(1)`.
    pub fn sum(&self) -> f64 {
        unsafe { igraph_psumtree_sum(self) }
    }

    /// Sets the weight of item `index` (`igraph_psumtree_update`). `O(log n)`.
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) if
    /// `index` is out of bounds or `weight` is negative, infinite or NaN.
    pub fn update(&mut self, index: usize, weight: f64) -> Result<()> {
        if index >= self.len() {
            return Err(Error::invalid(format!(
                "item {index} out of bounds for a tree of {} items",
                self.len()
            )));
        }
        igraph_call!(igraph_psumtree_update(self, index as igraph_int_t, weight))
    }

    /// Resets all the weights to zero (`igraph_psumtree_reset`).
    pub fn reset(&mut self) {
        unsafe { igraph_psumtree_reset(self) }
    }

    /// Finds the item whose cumulative weight interval contains `value`
    /// (`igraph_psumtree_search`). `O(log n)`.
    ///
    /// Precisely, it returns the lowest index `i` such that the total weight
    /// of the items before `i` is `<= value` and adding the weight of `i`
    /// makes it `> value`. Items of weight zero are therefore never
    /// returned.
    ///
    /// # Errors
    /// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) unless
    /// `0 <= value < self.sum()` (in particular when all weights are zero).
    pub fn search(&self, value: f64) -> Result<usize> {
        let sum = self.sum();
        if !(value >= 0.0 && value < sum) {
            return Err(Error::invalid(format!(
                "search value {value} is outside of [0, {sum})"
            )));
        }
        let mut idx: igraph_int_t = 0;
        igraph_call!(igraph_psumtree_search(self, &mut idx, value))?;
        // Guard against round-off landing on the zero padding of the tree.
        let idx = usize::try_from(idx).unwrap_or(usize::MAX);
        if idx >= self.len() {
            return Err(Error::invalid(format!(
                "search value {value} is too close to the total weight {sum}"
            )));
        }
        Ok(idx)
    }

    /// Draws an item with probability proportional to its weight, using the
    /// thread's default random number generator; `None` if all weights are
    /// zero. `O(log n)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use igraph::{misc::PsumTree, prelude::*};
    ///
    /// rng::seed(3)?;
    /// let tree = PsumTree::from_weights(&[0.0, 1.0, 0.0])?;
    /// assert!((0..100).all(|_| tree.sample() == Some(1)));
    /// # Ok::<(), igraph::Error>(())
    /// ```
    pub fn sample(&self) -> Option<usize> {
        let sum = self.sum();
        if sum.is_nan() || sum <= 0.0 {
            return None;
        }
        // Retry in the (extremely unlikely) case of round-off at the upper end.
        for _ in 0..64 {
            if let Ok(i) = self.search(crate::rng::uniform(0.0, sum)) {
                return Some(i);
            }
        }
        (0..self.len())
            .rev()
            .find(|&i| self.get(i).is_some_and(|w| w > 0.0))
    }
}

impl Drop for igraph_psumtree_t {
    fn drop(&mut self) {
        // `igraph_psumtree_destroy` frees the inner vector and nulls its
        // storage pointer, so the drop glue of the `v` field is then a no-op.
        if !self.v.stor_begin.is_null() {
            unsafe { igraph_psumtree_destroy(self) };
        }
    }
}

impl Clone for igraph_psumtree_t {
    fn clone(&self) -> Self {
        Self {
            v: self.v.clone(),
            size: self.size,
            offset: self.offset,
        }
    }
}

//! Linear sum assignment problem (`igraph_lsap.h`).

use crate::{
    error::{Error, Result},
    ffi::*,
    igraph_call,
    matrix::Matrix,
    vector::VectorInt,
};

/// Solves a balanced linear sum assignment problem with the Hungarian
/// method.
///
/// `cost` is a square `n × n` matrix: rows are *agents*, columns are
/// *tasks*, and `cost[(i, j)]` is the cost of agent `i` performing task `j`.
/// The result assigns one distinct task to each agent (`result[i]` is the
/// task of agent `i`, so the result is a permutation of `0..n`) so that the
/// total cost `Σ cost[(i, result[i])]` is minimal.
///
/// - To *maximize* a profit instead, negate the matrix.
/// - To forbid an assignment, give it a large finite cost (larger than
///   the sum of all the other costs): infinite and NaN costs are rejected.
/// - For an unbalanced problem with more agents than tasks, add dummy
///   tasks (columns) of zero cost, and symmetrically for more tasks than
///   agents.
///
/// This is the minimum weight perfect matching of the complete bipartite
/// graph between agents and tasks; for sparse bipartite graphs see
/// [`Graph::maximum_bipartite_matching`](crate::Graph::maximum_bipartite_matching)
/// (which *maximizes* the total weight of a maximum matching).
///
/// Binds `igraph_solve_lsap` (declared in `igraph_lsap.h`, not in the HTML
/// manual). Time complexity: `O(n³)`.
///
/// # Errors
/// [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue) if the
/// matrix is not square, or if a cost is NaN or infinite (checked in Rust:
/// igraph 1.0.0 and 1.0.1 would loop forever).
///
/// # Examples
///
/// The example of igraph's own test suite:
///
/// ```
/// use igraph::{misc, prelude::*};
///
/// let cost = Matrix::from_rows(&[
///     [9.0, 2.0, 7.0, 8.0],
///     [6.0, 4.0, 3.0, 7.0],
///     [5.0, 8.0, 1.0, 8.0],
///     [7.0, 6.0, 9.0, 4.0],
/// ])?;
/// let tasks = misc::solve_lsap(&cost)?;
/// assert_eq!(tasks, vec![1, 0, 2, 3]);
/// let total: f64 = tasks.iter().enumerate().map(|(i, &j)| cost[(i, j as usize)]).sum();
/// assert_eq!(total, 13.0);
/// # Ok::<(), igraph::Error>(())
/// ```
pub fn solve_lsap(cost: &Matrix) -> Result<Vec<i64>> {
    let n = cost.nrow();
    if cost.ncol() != n {
        return Err(Error::invalid(format!(
            "the cost matrix must be square, got {}x{}",
            n,
            cost.ncol()
        )));
    }
    if n == 0 {
        return Ok(Vec::new());
    }
    // The Hungarian method of igraph 1.0.0 and 1.0.1 loops forever on NaN or
    // infinite costs (src/misc/lsap.c is unchanged in 1.0.1).
    if !cost.as_slice().iter().all(|c| c.is_finite()) {
        return Err(Error::invalid("the costs must be finite numbers"));
    }
    let mut res = VectorInt::new();
    igraph_call!(igraph_solve_lsap(cost, n as igraph_int_t, &mut res))?;
    Ok(res.into())
}

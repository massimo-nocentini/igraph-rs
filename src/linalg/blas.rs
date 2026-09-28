//! The BLAS interface (`igraph_blas.h`).
//!
//! Degenerate (empty) dimensions are handled on the Rust side, because the
//! bundled BLAS terminates the process on invalid leading dimensions.

use super::to_c_int;
use crate::{
    error::{Error, Result},
    ffi::*,
    igraph_call,
    matrix::Matrix,
    vector::Vector,
};

fn dgemv_dims(transpose: bool, a: &Matrix, x: usize, y: usize) -> Result<()> {
    to_c_int(a.nrow(), "number of rows")?;
    to_c_int(a.ncol(), "number of columns")?;
    let (xin, yout) = if transpose {
        (a.nrow(), a.ncol())
    } else {
        (a.ncol(), a.nrow())
    };
    if x != xin || y != yout {
        return Err(Error::invalid(format!(
            "dgemv: a {:?} matrix{} needs x of length {xin} and y of length {yout}, got {x} and {y}",
            a.shape(),
            if transpose { " (transposed)" } else { "" }
        )));
    }
    Ok(())
}

/// Matrix-vector product `alpha * op(A) * x + beta * y`, where `op(A)` is
/// `A` or its transpose (`igraph_blas_dgemv`); the result is returned as a
/// new vector. When `beta` is zero, `y` only provides the length.
///
/// Time complexity: O(nk) for an `n` × `k` matrix.
///
/// Binds [`igraph_blas_dgemv`](https://igraph.org/c/html/latest/igraph-Linalg.html#igraph_blas_dgemv).
///
/// # Errors
/// If the lengths of `x` and `y` do not match `op(A)`.
///
/// # Examples
///
/// ```
/// use igraph::{linalg::blas_dgemv, prelude::*};
/// let a = Matrix::from_rows(&[[1.0, 2.0], [3.0, 4.0]]).unwrap();
/// assert_eq!(blas_dgemv(false, 1.0, &a, &[1.0, 1.0], 0.0, &[0.0, 0.0]).unwrap(), vec![3.0, 7.0]);
/// assert_eq!(blas_dgemv(true, 2.0, &a, &[1.0, 1.0], 1.0, &[1.0, 1.0]).unwrap(), vec![9.0, 13.0]);
/// ```
pub fn blas_dgemv(
    transpose: bool,
    alpha: f64,
    a: &Matrix,
    x: &[f64],
    beta: f64,
    y: &[f64],
) -> Result<Vec<f64>> {
    dgemv_dims(transpose, a, x.len(), y.len())?;
    let mut out = Vector::from_slice(y);
    if a.is_empty() {
        // op(A) x is an empty sum; BLAS would reject the zero leading dimension.
        for yi in out.iter_mut() {
            *yi = if beta == 0.0 { 0.0 } else { beta * *yi };
        }
        return Ok(out.into());
    }
    let xv = Vector::view(x);
    igraph_call!(igraph_blas_dgemv(
        transpose,
        alpha,
        a,
        xv.as_ptr(),
        beta,
        &mut out
    ))?;
    Ok(out.into())
}

/// In-place matrix-vector product `y <- alpha * op(A) * x + beta * y` on
/// plain slices (`igraph_blas_dgemv_array`).
///
/// Binds [`igraph_blas_dgemv_array`](https://igraph.org/c/html/latest/igraph-Linalg.html#igraph_blas_dgemv_array).
///
/// # Errors
/// If the lengths of `x` and `y` do not match `op(A)`.
pub fn blas_dgemv_array(
    transpose: bool,
    alpha: f64,
    a: &Matrix,
    x: &[f64],
    beta: f64,
    y: &mut [f64],
) -> Result<()> {
    dgemv_dims(transpose, a, x.len(), y.len())?;
    if a.is_empty() {
        // op(A) x is an empty sum; BLAS would reject the zero leading dimension.
        for yi in y.iter_mut() {
            *yi = if beta == 0.0 { 0.0 } else { beta * *yi };
        }
        return Ok(());
    }
    igraph_call!(igraph_blas_dgemv_array(
        transpose,
        alpha,
        a,
        x.as_ptr(),
        beta,
        y.as_mut_ptr()
    ))
}

/// Matrix-matrix product `alpha * op(A) * op(B) + beta * C`, where `op(X)`
/// is `X` or its transpose (`igraph_blas_dgemm`). `c` may be `None` when
/// `beta` is zero (or to mean a zero matrix).
///
/// igraph 1.0.0 and 1.0.1 check the shape of `C` against the wrong dimension when
/// `beta != 0`; this wrapper validates the shapes itself and adds `beta * C`
/// on the Rust side when igraph would reject a correct call.
///
/// Time complexity: O(nmk) for an `n` × `k` times `k` × `m` product.
///
/// Binds [`igraph_blas_dgemm`](https://igraph.org/c/html/latest/igraph-Linalg.html#igraph_blas_dgemm).
///
/// # Examples
///
/// ```
/// use igraph::{linalg::blas_dgemm, prelude::*};
/// let a = Matrix::from_rows(&[[1.0, 2.0, 3.0]]).unwrap(); // 1x3
/// let b = Matrix::from_rows(&[[1.0], [1.0], [1.0]]).unwrap(); // 3x1
/// let ab = blas_dgemm(false, false, 1.0, &a, &b, 0.0, None).unwrap();
/// assert_eq!(ab.to_rows(), vec![vec![6.0]]);
/// // The outer product B A is 3x3.
/// let ba = blas_dgemm(false, false, 1.0, &b, &a, 0.0, None).unwrap();
/// assert_eq!(ba.shape(), (3, 3));
/// // A' A with transposition flags.
/// let ata = blas_dgemm(true, false, 1.0, &a, &a, 0.0, None).unwrap();
/// assert_eq!(ata[(2, 1)], 6.0);
/// ```
pub fn blas_dgemm(
    transpose_a: bool,
    transpose_b: bool,
    alpha: f64,
    a: &Matrix,
    b: &Matrix,
    beta: f64,
    c: Option<&Matrix>,
) -> Result<Matrix> {
    for m in [a, b] {
        to_c_int(m.nrow(), "number of rows")?;
        to_c_int(m.ncol(), "number of columns")?;
    }
    let (m, k) = if transpose_a {
        (a.ncol(), a.nrow())
    } else {
        (a.nrow(), a.ncol())
    };
    let (kb, n) = if transpose_b {
        (b.ncol(), b.nrow())
    } else {
        (b.nrow(), b.ncol())
    };
    if k != kb {
        return Err(Error::invalid(format!(
            "dgemm: {m}-by-{k} and {kb}-by-{n} matrices cannot be multiplied"
        )));
    }
    let c = c.filter(|_| beta != 0.0);
    if let Some(c) = c
        && c.shape() != (m, n)
    {
        return Err(Error::invalid(format!(
            "dgemm: C is {:?}, expected {:?}",
            c.shape(),
            (m, n)
        )));
    }
    let mut res = Matrix::zeros(m, n);
    if m > 0 && n > 0 && k > 0 {
        // igraph 1.0.0 and 1.0.1 compare C's shape with (m, k) when beta != 0: only let it
        // add beta * C when that (buggy) check agrees with the real shape.
        if let Some(c) = c
            && k == n
        {
            res = c.clone();
            igraph_call!(igraph_blas_dgemm(
                transpose_a,
                transpose_b,
                alpha,
                a,
                b,
                beta,
                &mut res
            ))?;
            return Ok(res);
        }
        igraph_call!(igraph_blas_dgemm(
            transpose_a,
            transpose_b,
            alpha,
            a,
            b,
            0.0,
            &mut res
        ))?;
    }
    if let Some(c) = c {
        for (r, &cv) in res.as_mut_slice().iter_mut().zip(c.as_slice()) {
            *r += beta * cv;
        }
    }
    Ok(res)
}

/// Euclidean norm of a vector (`igraph_blas_dnrm2`), computed without
/// undue overflow or underflow.
///
/// Binds [`igraph_blas_dnrm2`](https://igraph.org/c/html/latest/igraph-Linalg.html#igraph_blas_dnrm2).
///
/// ```
/// use igraph::linalg::blas_dnrm2;
/// assert_eq!(blas_dnrm2(&[3.0, 4.0]).unwrap(), 5.0);
/// ```
pub fn blas_dnrm2(v: &[f64]) -> Result<f64> {
    to_c_int(v.len(), "vector length")?;
    crate::error::ensure_init();
    let view = Vector::view(v);
    Ok(unsafe { igraph_blas_dnrm2(view.as_ptr()) })
}

/// Dot product of two vectors of the same length (`igraph_blas_ddot`).
///
/// Binds [`igraph_blas_ddot`](https://igraph.org/c/html/latest/igraph-Linalg.html#igraph_blas_ddot).
///
/// ```
/// use igraph::linalg::blas_ddot;
/// assert_eq!(blas_ddot(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]).unwrap(), 32.0);
/// ```
///
/// # Errors
/// If the lengths differ.
pub fn blas_ddot(a: &[f64], b: &[f64]) -> Result<f64> {
    to_c_int(a.len(), "vector length")?;
    if a.len() != b.len() {
        return Err(Error::invalid(
            "dot product of vectors with different lengths",
        ));
    }
    let (va, vb) = (Vector::view(a), Vector::view(b));
    let mut res = 0.0;
    igraph_call!(igraph_blas_ddot(va.as_ptr(), vb.as_ptr(), &mut res))?;
    Ok(res)
}

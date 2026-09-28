//! Tests of the linear algebra module: sparse matrices, ARPACK, eigen,
//! LAPACK, BLAS and spectral embeddings.

mod common;

use common::*;
use igraph::linalg::*;
use igraph::prelude::*;
use std::f64::consts::PI;
use std::{
    cell::{Cell, RefCell},
    rc::Rc,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Dense Laplacian `D - A` of an undirected graph given by its edges.
fn laplacian_triplets(n: usize, edges: &[(usize, usize)]) -> Vec<(usize, usize, f64)> {
    let mut t = Vec::new();
    for &(a, b) in edges {
        t.extend([(a, b, -1.0), (b, a, -1.0), (a, a, 1.0), (b, b, 1.0)]);
    }
    let _ = n;
    t
}

fn path_edges(n: usize) -> Vec<(usize, usize)> {
    (0..n - 1).map(|i| (i, i + 1)).collect()
}

/// Eigenvalues of the path Laplacian on `n` vertices, increasing.
fn path_laplacian_spectrum(n: usize) -> Vec<f64> {
    (0..n)
        .map(|k| 2.0 - 2.0 * (PI * k as f64 / n as f64).cos())
        .collect()
}

fn path_laplacian(n: usize) -> SparseMat {
    SparseMat::from_triplets(n, n, &laplacian_triplets(n, &path_edges(n)))
        .unwrap()
        .compress()
        .unwrap()
}

/// Matrix-free product with the Laplacian of the path on `n` vertices.
fn path_laplacian_matvec(n: usize) -> impl FnMut(&[f64], &mut [f64]) {
    move |x: &[f64], y: &mut [f64]| {
        for i in 0..n {
            let mut deg = 0.0;
            if i > 0 {
                y[i] -= x[i - 1];
                deg += 1.0;
            }
            if i + 1 < n {
                y[i] -= x[i + 1];
                deg += 1.0;
            }
            y[i] += deg * x[i];
        }
    }
}

fn dense(rows: &[&[f64]]) -> Matrix {
    Matrix::from_rows(rows).unwrap()
}

fn assert_vec_close(a: &[f64], b: &[f64], eps: f64) {
    assert_eq!(a.len(), b.len(), "{a:?} vs {b:?}");
    for (x, y) in a.iter().zip(b) {
        assert_close(*x, *y, eps);
    }
}

fn assert_matrix_close(a: &Matrix, b: &Matrix, eps: f64) {
    assert_eq!(a.shape(), b.shape());
    assert_vec_close(a.as_slice(), b.as_slice(), eps);
}

fn sorted(mut v: Vec<f64>) -> Vec<f64> {
    v.sort_by(f64::total_cmp);
    v
}

/// Naive dense product, for cross-checking.
fn matmul(a: &Matrix, b: &Matrix) -> Matrix {
    let mut c = Matrix::zeros(a.nrow(), b.ncol());
    for i in 0..a.nrow() {
        for j in 0..b.ncol() {
            c[(i, j)] = (0..a.ncol()).map(|k| a[(i, k)] * b[(k, j)]).sum();
        }
    }
    c
}

/// A reproducible pseudo-random dense matrix.
fn random_matrix(nrow: usize, ncol: usize, seed: u64) -> Matrix {
    rng::seed(seed).unwrap();
    let data: Vec<f64> = (0..nrow * ncol).map(|_| rng::uniform(-1.0, 1.0)).collect();
    Matrix::from_column_major(nrow, ncol, &data).unwrap()
}

/// The Petersen graph: adjacency spectrum 3, 1 (x5), -2 (x4).
fn petersen() -> Graph {
    Graph::famous("Petersen").unwrap()
}

// ---------------------------------------------------------------------------
// Sparse matrices: construction and formats
// ---------------------------------------------------------------------------

#[test]
fn sparse_triplets_sum_duplicates_and_compress() {
    let mut m =
        SparseMat::from_triplets(3, 4, &[(0, 0, 1.0), (2, 3, 5.0), (0, 0, 2.0), (1, 2, -1.0)])
            .unwrap();
    assert!(m.is_triplet());
    assert_eq!(m.sparse_type(), SparseMatType::Triplet);
    assert_eq!(m.shape(), (3, 4));
    assert_eq!(m.nonzero_storage(), 4);
    assert_eq!(m.get(0, 0), 3.0);
    assert_eq!(m.get(2, 3), 5.0);
    assert_eq!(m.get(1, 1), 0.0);
    assert_eq!(m.get(10, 10), 0.0, "out of bounds reads are zero");

    let c = m.compress().unwrap();
    assert!(c.is_cc());
    assert_eq!(c.sparse_type(), SparseMatType::ColumnCompressed);
    assert_eq!(c.get(0, 0), 3.0);
    assert_eq!(c.to_dense().unwrap(), m.to_dense().unwrap());

    // dupl merges the duplicates (compressing `m` in place).
    m.dupl().unwrap();
    assert!(m.is_cc());
    assert_eq!(m.nonzero_storage(), 3);
    assert_eq!(
        m.to_dense().unwrap().to_rows(),
        vec![
            vec![3.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, -1.0, 0.0],
            vec![0.0, 0.0, 0.0, 5.0]
        ]
    );
}

#[test]
fn sparse_matrices_round_trip_through_graphs() {
    // Graph -> sparse adjacency (conversion) -> SparseMat -> graph
    // (constructors): the degree sequence is A 1, the number of closed
    // walks of length 2 is trace(A^2) = 2|E|, and the round trip is exact.
    let g = Graph::famous("Zachary").unwrap();
    let n = g.vcount();
    let coo = g
        .get_adjacency_sparse(GetAdjacency::Both, None, Loops::Twice)
        .unwrap();
    let triplets: Vec<(usize, usize, f64)> = coo
        .iter()
        .map(|(i, j, x)| (i as usize, j as usize, x))
        .collect();
    let a = SparseMat::from_triplets(n, n, &triplets)
        .unwrap()
        .compress()
        .unwrap();
    assert!(a.is_symmetric().unwrap());
    let degrees: Vec<f64> = g
        .degree(.., NeighborMode::All, Loops::Twice)
        .unwrap()
        .iter()
        .map(|&d| d as f64)
        .collect();
    assert_eq!(a.mul_vec(&vec![1.0; n]).unwrap(), degrees);
    assert_eq!(a.rowsums().unwrap(), degrees);
    let a2 = a.multiply(&a).unwrap();
    let trace: f64 = (0..n).map(|i| a2.get(i, i)).sum();
    assert_eq!(trace, 2.0 * g.ecount() as f64);

    let entries: Vec<(i64, i64, f64)> = a
        .triplets()
        .into_iter()
        .map(|(i, j, x)| (i as i64, j as i64, x))
        .collect();
    let back = Graph::sparse_adjacency(n, &entries, Adjacency::Undirected, Loops::Twice).unwrap();
    assert_eq!(back.vcount(), n);
    let normalized = |g: &Graph| {
        let mut e: Vec<(i64, i64)> = g
            .edge_list()
            .iter()
            .map(|&(a, b)| (a.min(b), a.max(b)))
            .collect();
        e.sort_unstable();
        e
    };
    assert_eq!(normalized(&back), normalized(&g));
}

#[test]
fn sparse_constructors() {
    let e = SparseMat::eye(4, 2.5, false).unwrap();
    assert!(e.is_triplet());
    assert_eq!(e.to_dense().unwrap()[(3, 3)], 2.5);
    let i = SparseMat::identity(3).unwrap();
    assert!(i.is_cc());
    assert_eq!(i.triplets(), vec![(0, 0, 1.0), (1, 1, 1.0), (2, 2, 1.0)]);

    let d = SparseMat::diag(&[1.0, -2.0, 3.0], false).unwrap();
    assert_eq!(d.rowsums().unwrap(), vec![1.0, -2.0, 3.0]);
    let dc = SparseMat::diag(&[1.0, -2.0, 3.0], true).unwrap();
    assert_eq!(dc.colsums().unwrap(), vec![1.0, -2.0, 3.0]);

    let empty = SparseMat::new(0, 0).unwrap();
    assert_eq!(empty.shape(), (0, 0));
    assert_eq!(empty.iter().count(), 0);
    let with_cap = SparseMat::with_capacity(5, 5, 100).unwrap();
    assert!(with_cap.nzmax() >= 100);

    // Dense round trip with tolerance.
    let m = dense(&[&[1.0, 1e-10, 0.0], &[0.0, -3.0, 2.0]]);
    let s = SparseMat::from_dense(&m, 1e-8).unwrap();
    assert_eq!(s.nonzero_storage(), 3);
    let back: Matrix = Matrix::try_from(&s).unwrap();
    assert_eq!(
        back.to_rows(),
        vec![vec![1.0, 0.0, 0.0], vec![0.0, -3.0, 2.0]]
    );
    let s2 = SparseMat::try_from(&m).unwrap();
    assert_eq!(s2.nonzero_storage(), 4);
}

#[test]
fn sparse_entry_errors() {
    let mut m = SparseMat::new(2, 2).unwrap();
    m.entry(1, 1, 1.0).unwrap();
    assert_eq!(
        m.entry(2, 0, 1.0).unwrap_err().kind(),
        ErrorKind::InvalidValue
    );
    assert!(SparseMat::from_triplets(2, 2, &[(0, 5, 1.0)]).is_err());
    let mut c = m.compress().unwrap();
    // Entries can only be added to triplet matrices.
    assert_eq!(
        c.entry(0, 0, 1.0).unwrap_err().kind(),
        ErrorKind::InvalidValue
    );
}

#[test]
fn sparse_transpose_both_formats() {
    let t = SparseMat::from_triplets(2, 3, &[(0, 1, 1.0), (1, 2, 2.0), (0, 0, 3.0)]).unwrap();
    let expected = t.to_dense().unwrap().transposed();
    // Non-square triplet matrices: igraph forgets to swap the dimensions,
    // the wrapper handles it.
    let tt = t.transpose().unwrap();
    assert_eq!(tt.shape(), (3, 2));
    assert!(tt.is_triplet());
    assert_eq!(tt.to_dense().unwrap(), expected);
    let ct = t.compress().unwrap().transpose().unwrap();
    assert!(ct.is_cc());
    assert_eq!(ct.to_dense().unwrap(), expected);
    // Square triplet matrix goes through igraph directly.
    let sq = SparseMat::from_triplets(2, 2, &[(0, 1, 7.0)]).unwrap();
    assert_eq!(sq.transpose().unwrap().get(1, 0), 7.0);
    // Transposing twice is the identity.
    assert_eq!(
        ct.transpose().unwrap().to_dense().unwrap(),
        t.to_dense().unwrap()
    );
}

#[test]
fn sparse_symmetry() {
    assert!(path_laplacian(6).is_symmetric().unwrap());
    let t = SparseMat::from_triplets(3, 3, &[(0, 1, 1.0), (1, 0, 1.0), (2, 2, 4.0)]).unwrap();
    assert!(t.is_symmetric().unwrap());
    let a = SparseMat::from_triplets(3, 3, &[(0, 1, 1.0)]).unwrap();
    assert!(!a.is_symmetric().unwrap());
    assert!(!SparseMat::new(2, 3).unwrap().is_symmetric().unwrap());
}

// ---------------------------------------------------------------------------
// Sparse arithmetic
// ---------------------------------------------------------------------------

#[test]
fn sparse_arithmetic_matches_dense() {
    let a = SparseMat::from_dense(&random_matrix(4, 3, 1), 0.3).unwrap();
    let b = SparseMat::from_dense(&random_matrix(3, 5, 2), 0.3).unwrap();
    let (ad, bd) = (a.to_dense().unwrap(), b.to_dense().unwrap());

    let ab = (&a * &b).unwrap();
    assert!(ab.is_cc());
    assert_matrix_close(&ab.to_dense().unwrap(), &matmul(&ad, &bd), 1e-12);
    assert_matrix_close(&a.multiply_by_dense(&bd).unwrap(), &matmul(&ad, &bd), 1e-12);
    assert_matrix_close(&dense_multiply(&ad, &b).unwrap(), &matmul(&ad, &bd), 1e-12);

    let sum = (&a + &a).unwrap();
    let diff = (&a - &a).unwrap();
    let combo = a.add(&a, 2.0, -0.5).unwrap();
    for i in 0..4 {
        for j in 0..3 {
            assert_close(sum.get(i, j), 2.0 * ad[(i, j)], 1e-12);
            assert_close(diff.get(i, j), 0.0, 1e-12);
            assert_close(combo.get(i, j), 1.5 * ad[(i, j)], 1e-12);
        }
    }

    // Matrix-vector products.
    let x = [1.0, -2.0, 0.5];
    let y = [1.0, 1.0, 1.0, 1.0];
    let ax = a.mul_vec(&x).unwrap();
    let naive: Vec<f64> = (0..4)
        .map(|i| (0..3).map(|k| ad[(i, k)] * x[k]).sum())
        .collect();
    assert_vec_close(&ax, &naive, 1e-12);
    let g = a.gaxpy(&x, &y).unwrap();
    assert_vec_close(
        &g,
        &naive.iter().map(|v| v + 1.0).collect::<Vec<_>>(),
        1e-12,
    );

    // Shape errors.
    assert_eq!((&a * &a).unwrap_err().kind(), ErrorKind::InvalidValue);
    assert!((&a + &b).is_err());
    assert!(a.mul_vec(&[1.0]).is_err());
    assert!(a.multiply_by_dense(&ad).is_err());
    assert!(dense_multiply(&bd, &b).is_err());
}

#[test]
fn sparse_scaling_and_normalization() {
    let mut m = SparseMat::from_triplets(2, 3, &[(0, 0, 1.0), (0, 2, 3.0), (1, 1, 2.0)]).unwrap();
    m.scale(2.0).unwrap();
    assert_eq!(m.get(0, 2), 6.0);
    m.scale_rows(&[1.0, 0.5]).unwrap();
    assert_eq!(m.get(1, 1), 2.0);
    m.scale_cols(&[1.0, 1.0, 0.5]).unwrap();
    assert_eq!(m.get(0, 2), 3.0);
    m.neg().unwrap();
    assert_eq!(
        m.to_dense().unwrap().to_rows(),
        vec![vec![-2.0, 0.0, -3.0], vec![0.0, -2.0, 0.0]]
    );
    assert!(m.scale_rows(&[1.0]).is_err());
    assert!(m.scale_cols(&[1.0]).is_err());

    // A random walk transition matrix: rows of D^-1 A sum to one.
    let mut walk =
        SparseMat::from_triplets(3, 3, &[(0, 1, 1.0), (0, 2, 3.0), (1, 0, 2.0), (2, 1, 5.0)])
            .unwrap();
    walk.normalize_rows(false).unwrap();
    assert_vec_close(&walk.rowsums().unwrap(), &[1.0, 1.0, 1.0], 1e-12);
    assert_close(walk.get(0, 2), 0.75, 1e-12);

    let mut cols =
        SparseMat::from_triplets(2, 2, &[(0, 0, 1.0), (1, 0, 3.0), (0, 1, 2.0)]).unwrap();
    cols.normalize_cols(false).unwrap();
    assert_vec_close(&cols.colsums().unwrap(), &[1.0, 1.0], 1e-12);

    let mut zero_row = SparseMat::from_triplets(2, 2, &[(0, 0, 1.0)]).unwrap();
    assert_eq!(
        zero_row.normalize_rows(false).unwrap_err().kind(),
        ErrorKind::InvalidValue
    );
    zero_row.normalize_rows(true).unwrap();
    assert!(SparseMat::new(2, 3).unwrap().normalize_cols(true).is_err());
}

#[test]
fn sparse_filters() {
    let mut m = SparseMat::from_dense(
        &dense(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], &[7.0, 8.0, 9.0]]),
        0.0,
    )
    .unwrap();
    // Keep the strictly lower triangle.
    m.fkeep(|i, j, _| i > j).unwrap();
    assert!(m.is_cc());
    assert_eq!(
        sorted(m.triplets().iter().map(|t| t.2).collect()),
        vec![4.0, 7.0, 8.0]
    );

    let mut z = SparseMat::from_triplets(2, 2, &[(0, 0, 0.0), (1, 1, 1e-9), (0, 1, 1.0)]).unwrap();
    z.dropzeros().unwrap();
    assert_eq!(z.nonzero_storage(), 2);
    z.droptol(1e-6).unwrap();
    assert_eq!(z.nonzero_storage(), 1);
    assert_eq!(z.get(0, 1), 1.0);

    let mut small =
        SparseMat::from_triplets(2, 2, &[(0, 0, 1e-9), (1, 1, 1.0), (0, 1, -0.5)]).unwrap();
    assert_eq!(small.count_nonzero().unwrap(), 3);
    assert_eq!(small.count_nonzerotol(1e-6).unwrap(), 2);
}

#[test]
fn sparse_fkeep_panic_is_resumed() {
    let mut m = SparseMat::identity(3).unwrap();
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        m.fkeep(|_, _, _| panic!("boom")).unwrap();
    }));
    assert!(r.is_err());
    // The matrix is still usable.
    assert_eq!(m.shape(), (3, 3));
}

#[test]
fn sparse_min_max_statistics() {
    // The largest element is stored last: igraph 1.0.0 and 1.0.1's `igraph_sparsemat_max`
    // would miss it, the wrapper does not.
    let mut m = SparseMat::from_triplets(3, 3, &[(0, 0, -1.0), (1, 0, 2.0), (2, 2, 7.0)]).unwrap();
    assert_eq!(m.max().unwrap(), 7.0);
    let mut m2 = SparseMat::from_triplets(3, 3, &[(0, 0, 4.0), (1, 1, 2.0), (2, 2, -7.0)]).unwrap();
    assert_eq!(m2.min().unwrap(), -7.0);
    assert_eq!(m2.minmax().unwrap(), (-7.0, 4.0));
    let mut empty = SparseMat::new(2, 2).unwrap();
    assert_eq!(empty.max().unwrap(), f64::NEG_INFINITY);
    assert_eq!(empty.min().unwrap(), f64::INFINITY);

    // Row/column extrema only consider stored values.
    let mut r = SparseMat::from_triplets(3, 2, &[(0, 0, 3.0), (0, 1, -1.0), (2, 1, 5.0)]).unwrap();
    assert_eq!(r.rowmins().unwrap(), vec![-1.0, f64::INFINITY, 5.0]);
    assert_eq!(r.rowmaxs().unwrap(), vec![3.0, f64::NEG_INFINITY, 5.0]);
    assert_eq!(r.colmins().unwrap(), vec![3.0, -1.0]);
    assert_eq!(r.colmaxs().unwrap(), vec![3.0, 5.0]);
    let (vals, pos) = r.which_min_rows().unwrap();
    assert_eq!(vals, vec![-1.0, f64::INFINITY, 5.0]);
    assert_eq!(pos[0], 1);
    assert_eq!(pos[2], 1);
    let mut rc = r.compress().unwrap();
    let (vals, pos) = rc.which_min_cols().unwrap();
    assert_eq!(vals, vec![3.0, -1.0]);
    assert_eq!(pos, vec![0, 0]);
    assert_eq!(rc.rowmins().unwrap(), vec![-1.0, f64::INFINITY, 5.0]);
}

#[test]
fn sparse_elements_iteration_and_sorting() {
    let t = SparseMat::from_triplets(3, 3, &[(2, 1, 1.0), (0, 1, 2.0), (1, 0, 3.0), (0, 2, 4.0)])
        .unwrap();
    let e = t.getelements().unwrap();
    assert_eq!(e.i, vec![2, 0, 1, 0]);
    assert_eq!(e.j, vec![1, 1, 0, 2]);
    assert_eq!(e.x, vec![1.0, 2.0, 3.0, 4.0]);

    // Sorted by column, then row.
    let s = t.getelements_sorted().unwrap();
    assert_eq!(s.i, vec![1, 0, 2, 0]);
    assert_eq!(s.j, vec![0, 1, 1, 2]);
    assert_eq!(s.x, vec![3.0, 2.0, 1.0, 4.0]);
    let st = t.sort().unwrap();
    assert!(st.is_triplet());
    assert_eq!(
        st.triplets(),
        vec![(1, 0, 3.0), (0, 1, 2.0), (2, 1, 1.0), (0, 2, 4.0)]
    );

    // Column-compressed layout: j holds the column pointers.
    let c = t.compress().unwrap().sort().unwrap();
    let ce = c.getelements().unwrap();
    assert_eq!(ce.j, vec![0, 1, 3, 4]);
    assert_eq!(ce.i, vec![1, 0, 2, 0]);

    // The iterator visits every stored entry; reset restarts it.
    let mut it = c.iter();
    assert_eq!(it.index(), 0);
    let all: Vec<_> = it.by_ref().collect();
    assert_eq!(all, c.triplets());
    assert_eq!(it.next(), None);
    it.reset();
    assert_eq!(it.next(), Some((1, 0, 3.0)));
    assert_eq!(it.index(), 1);

    // Empty columns are skipped.
    let gaps = SparseMat::from_triplets(2, 4, &[(1, 3, 9.0)])
        .unwrap()
        .compress()
        .unwrap();
    assert_eq!(gaps.triplets(), vec![(1, 3, 9.0)]);
}

#[test]
fn sparse_triplet_iteration_with_tight_storage() {
    // A triplet matrix whose capacity (nzmax = 1) is smaller than its
    // number of columns: igraph's own iterator would read past the column
    // index array here, the wrapper walks a copy of the entries instead.
    let m = SparseMat::from_triplets(2, 6, &[(1, 5, 2.5)]).unwrap();
    assert_eq!(m.nzmax(), 1);
    assert_eq!(m.triplets(), vec![(1, 5, 2.5)]);
    assert_eq!(m.get(1, 5), 2.5);
    assert_eq!(m.get(0, 5), 0.0);
    assert_eq!(m.get(1, 4), 0.0);
    let mut it = m.iter();
    assert_eq!(it.next(), Some((1, 5, 2.5)));
    assert_eq!(it.index(), 1);
    assert_eq!(it.next(), None);
    it.reset();
    assert_eq!(it.index(), 0);
    assert_eq!(it.count(), 1);
    // Duplicates are summed by `get`, but visited separately by `iter`.
    let mut d = SparseMat::new(1, 3).unwrap();
    d.entry(0, 2, 1.0).unwrap();
    d.entry(0, 2, 2.0).unwrap();
    assert_eq!(d.get(0, 2), 3.0);
    assert_eq!(d.iter().count(), 2);
    // Transposing the non-square triplet matrix goes through the iterator too.
    let t = m.transpose().unwrap();
    assert_eq!(t.shape(), (6, 2));
    assert_eq!(t.get(5, 1), 2.5);
    // An empty triplet matrix has nothing to visit.
    assert_eq!(SparseMat::new(3, 7).unwrap().iter().next(), None);
}

#[test]
fn sparse_permute_and_index() {
    let m = SparseMat::from_dense(
        &dense(&[&[1.0, 2.0, 0.0], &[0.0, 3.0, 4.0], &[5.0, 0.0, 6.0]]),
        0.0,
    )
    .unwrap();
    // Reverse the rows, rotate the columns.
    let p = m.permute(&[2, 1, 0], &[1, 2, 0]).unwrap();
    let d = m.to_dense().unwrap();
    let pd = p.to_dense().unwrap();
    for i in 0..3 {
        for (j, &q) in [1, 2, 0].iter().enumerate() {
            assert_eq!(pd[(i, j)], d[(2 - i, q)]);
        }
    }
    assert!(m.permute(&[0, 0, 1], &[0, 1, 2]).is_err());
    assert!(m.permute(&[0, 1], &[0, 1, 2]).is_err());

    let sub = m.index(Some(&[0, 2]), Some(&[2, 0])).unwrap();
    assert_eq!(
        sub.to_dense().unwrap().to_rows(),
        vec![vec![0.0, 1.0], vec![6.0, 5.0]]
    );
    let rows = m.index(Some(&[1]), None).unwrap();
    assert_eq!(
        rows.to_dense().unwrap().to_rows(),
        vec![vec![0.0, 3.0, 4.0]]
    );
    let cols = m.index(None, Some(&[1, 1])).unwrap();
    assert_eq!(cols.to_dense().unwrap().column(0), &[2.0, 3.0, 0.0]);
    assert_eq!(cols.to_dense().unwrap().column(1), &[2.0, 3.0, 0.0]);
    assert_eq!(m.index(None, None).unwrap().to_dense().unwrap(), d);
    assert!(m.index(Some(&[3]), None).is_err());
}

#[test]
fn sparse_resizing() {
    let mut m = SparseMat::from_triplets(2, 2, &[(0, 0, 1.0), (1, 1, 2.0)]).unwrap();
    m.add_rows(1).unwrap();
    m.add_cols(2).unwrap();
    assert_eq!(m.shape(), (3, 4));
    m.entry(2, 3, 5.0).unwrap();
    assert_eq!(m.get(2, 3), 5.0);
    let mut c = m.compress().unwrap();
    c.add_cols(1).unwrap();
    assert_eq!(c.shape(), (3, 5));
    assert_eq!(c.colsums().unwrap(), vec![1.0, 2.0, 0.0, 5.0, 0.0]);
    c.realloc(100).unwrap();
    assert!(c.nzmax() >= 100);
    c.resize(4, 4, 10).unwrap();
    assert_eq!(c.shape(), (4, 4));
    assert!(c.is_triplet());
    assert_eq!(c.nonzero_storage(), 0);
}

#[test]
fn sparse_printing() {
    let t = SparseMat::from_triplets(2, 2, &[(0, 1, 1.5), (1, 0, -2.0)]).unwrap();
    assert_eq!(t.print_to_string().unwrap(), "0 1 : 1.5\n1 0 : -2\n");
    let c = t.compress().unwrap();
    let s = format!("{c}");
    assert!(s.starts_with("col 0: locations 0 to 0\n1 : -2\n"), "{s}");
    assert!(s.contains("col 1: locations 1 to 1\n0 : 1.5\n"), "{s}");
}

#[test]
fn sparse_clone_is_deep() {
    let a = SparseMat::from_triplets(2, 2, &[(0, 0, 1.0)]).unwrap();
    let mut b = a.clone();
    b.entry(1, 1, 3.0).unwrap();
    assert_eq!(a.nonzero_storage(), 1);
    assert_eq!(b.nonzero_storage(), 2);
    // Matrices can be sent to other threads.
    let h = std::thread::spawn(move || b.get(1, 1));
    assert_eq!(h.join().unwrap(), 3.0);
}

// ---------------------------------------------------------------------------
// Sparse solvers
// ---------------------------------------------------------------------------

#[test]
fn sparse_triangular_solvers() {
    // Lower triangular L with entries added out of order (so igraph's
    // compressed columns are unsorted); the wrapper sorts them.
    let l = SparseMat::from_triplets(
        3,
        3,
        &[
            (2, 0, 1.0),
            (1, 1, 3.0),
            (0, 0, 2.0),
            (1, 0, 1.0),
            (2, 2, 4.0),
            (2, 1, -1.0),
        ],
    )
    .unwrap();
    let ld = l.to_dense().unwrap();
    let x = [1.0, -1.0, 2.0];
    let b: Vec<f64> = (0..3)
        .map(|i| (0..3).map(|k| ld[(i, k)] * x[k]).sum())
        .collect();
    assert_vec_close(&l.lsolve(&b).unwrap(), &x, 1e-12);
    let bt: Vec<f64> = (0..3)
        .map(|i| (0..3).map(|k| ld[(k, i)] * x[k]).sum())
        .collect();
    assert_vec_close(&l.ltsolve(&bt).unwrap(), &x, 1e-12);

    let u = l.transpose().unwrap();
    assert_vec_close(&u.usolve(&bt).unwrap(), &x, 1e-12);
    assert_vec_close(&u.utsolve(&b).unwrap(), &x, 1e-12);

    // Wrong structure is rejected instead of reading garbage.
    assert_eq!(u.lsolve(&b).unwrap_err().kind(), ErrorKind::InvalidValue);
    assert!(l.usolve(&b).is_err());
    let no_diag = SparseMat::from_triplets(2, 2, &[(1, 0, 1.0), (1, 1, 1.0)]).unwrap();
    assert!(no_diag.lsolve(&[1.0, 1.0]).is_err());
    assert!(l.lsolve(&[1.0]).is_err());
    assert!(SparseMat::new(2, 3).unwrap().lsolve(&[1.0, 1.0]).is_err());
}

#[test]
fn sparse_cholesky_lu_qr_solve_the_same_system() {
    // L + I for the path on 30 vertices: symmetric positive definite.
    let n = 30;
    let a = path_laplacian(n)
        .add(&SparseMat::identity(n).unwrap(), 1.0, 1.0)
        .unwrap();
    let x: Vec<f64> = (0..n).map(|i| (i as f64).sin()).collect();
    let b = a.mul_vec(&x).unwrap();

    for order in [SparseOrdering::Natural, SparseOrdering::MinDegreeSymmetric] {
        assert_vec_close(&a.cholsol(&b, order).unwrap(), &x, 1e-10);
    }
    for order in [
        SparseOrdering::Natural,
        SparseOrdering::MinDegreeSymmetric,
        SparseOrdering::MinDegreeNoDenseRows,
        SparseOrdering::MinDegreeAtA,
    ] {
        assert_vec_close(&a.lusol(&b, order, 1.0).unwrap(), &x, 1e-10);
        let lu = a.lu(order, 1.0).unwrap();
        assert_eq!(lu.dim(), n);
        assert_vec_close(&lu.solve(&b).unwrap(), &x, 1e-10);
        let qr = a.qr(order).unwrap();
        assert_eq!(qr.dim(), n);
        assert_vec_close(&qr.solve(&b).unwrap(), &x, 1e-10);
    }

    // One factorization, many right hand sides.
    let lu = a.lu(SparseOrdering::MinDegreeSymmetric, 0.001).unwrap();
    for k in 0..5 {
        let xk: Vec<f64> = (0..n).map(|i| (i * k) as f64).collect();
        let bk = a.mul_vec(&xk).unwrap();
        assert_vec_close(&lu.solve(&bk).unwrap(), &xk, 1e-9);
    }
    assert!(lu.solve(&[1.0]).is_err());
}

#[test]
fn sparse_solver_failures() {
    // The Laplacian itself is singular (constant vectors are in its kernel)...
    let l = path_laplacian(5);
    let b = vec![1.0; 5];
    assert!(
        l.lu(SparseOrdering::Natural, 1.0).is_err()
            || l.lusol(&b, SparseOrdering::Natural, 1.0).is_err()
    );
    // ... and not positive definite.
    let neg = SparseMat::diag(&[1.0, -1.0], true).unwrap();
    assert!(neg.cholsol(&[1.0, 1.0], SparseOrdering::Natural).is_err());
    // Non-square input.
    let rect = SparseMat::new(2, 3).unwrap();
    assert_eq!(
        rect.lusol(&[1.0, 1.0], SparseOrdering::Natural, 1.0)
            .unwrap_err()
            .kind(),
        ErrorKind::InvalidValue
    );
    assert!(rect.lu(SparseOrdering::Natural, 1.0).is_err());
    assert!(rect.qr(SparseOrdering::Natural).is_err());
    assert!(
        path_laplacian(3)
            .cholsol(&[1.0], SparseOrdering::Natural)
            .is_err()
    );
}

// ---------------------------------------------------------------------------
// ARPACK
// ---------------------------------------------------------------------------

#[test]
fn arpack_path_laplacian_matches_analytic_spectrum() {
    rng::seed(42).unwrap();
    let n = 50;
    let spectrum = path_laplacian_spectrum(n);

    // Largest algebraic eigenvalues come in decreasing order.
    let opts = ArpackOptions::default()
        .with_nev(4)
        .with_which(ArpackWhich::LargestAlgebraic);
    let res = arpack_rssolve(n, path_laplacian_matvec(n), &opts, None).unwrap();
    let expected: Vec<f64> = spectrum.iter().rev().take(4).copied().collect();
    assert_vec_close(&res.values, &expected, 1e-8);
    assert_eq!(res.vectors.shape(), (n, 4));
    assert!(res.info.iterations > 0);
    assert!(res.info.nconv >= 4);

    // Smallest algebraic ones, in increasing order: 0 first.
    let opts = ArpackOptions::default()
        .with_nev(3)
        .with_which(ArpackWhich::SmallestAlgebraic)
        .with_ncv(20);
    let res = arpack_rssolve(n, path_laplacian_matvec(n), &opts, None).unwrap();
    assert_vec_close(&sorted(res.values.clone()), &spectrum[..3], 1e-8);
    // The eigenvectors are unit vectors satisfying L v = lambda v.
    for k in 0..3 {
        let v = res.vectors.column(k);
        assert_close(blas_dnrm2(v).unwrap(), 1.0, 1e-10);
        let mut lv = vec![0.0; n];
        path_laplacian_matvec(n)(v, &mut lv);
        for i in 0..n {
            assert_close(lv[i], res.values[k] * v[i], 1e-7);
        }
    }

    // Both ends: 2 from the top, 1 from the bottom.
    let opts = ArpackOptions::default()
        .with_nev(3)
        .with_which(ArpackWhich::BothEnds);
    let res = arpack_rssolve(n, path_laplacian_matvec(n), &opts, None).unwrap();
    let got = sorted(res.values);
    assert_close(got[0], 0.0, 1e-8);
    assert_close(got[1], spectrum[n - 2], 1e-8);
    assert_close(got[2], spectrum[n - 1], 1e-8);
}

#[test]
fn arpack_storage_start_vector_and_small_problems() {
    let mut storage = ArpackStorage::new(40, 20, true).unwrap();
    assert_eq!((storage.maxn(), storage.maxncv()), (40, 20));
    assert!(storage.is_symmetric());
    for n in [10, 25, 40] {
        let opts = ArpackOptions::default()
            .with_nev(1)
            .with_which(ArpackWhich::LargestAlgebraic)
            .with_ncv(10);
        let res = arpack_rssolve(n, path_laplacian_matvec(n), &opts, Some(&mut storage)).unwrap();
        assert_close(
            res.values[0],
            *path_laplacian_spectrum(n).last().unwrap(),
            1e-8,
        );
    }
    // Too small a storage is an error, not a buffer overflow.
    let mut tiny = ArpackStorage::new(5, 3, true).unwrap();
    let opts = ArpackOptions::default().with_nev(1).with_ncv(4);
    assert!(arpack_rssolve(10, path_laplacian_matvec(10), &opts, Some(&mut tiny)).is_err());
    assert!(ArpackStorage::new(0, 3, true).is_err());

    // With a given start vector the result does not depend on the RNG.
    let n = 20;
    let start: Vec<f64> = (0..n).map(|i| 1.0 + i as f64).collect();
    let opts = ArpackOptions::default()
        .with_which(ArpackWhich::LargestAlgebraic)
        .with_start(start);
    rng::seed(1).unwrap();
    let r1 = arpack_rssolve(n, path_laplacian_matvec(n), &opts, None).unwrap();
    rng::seed(2).unwrap();
    let r2 = arpack_rssolve(n, path_laplacian_matvec(n), &opts, None).unwrap();
    assert_eq!(r1, r2);
    assert_close(
        r1.values[0],
        *path_laplacian_spectrum(n).last().unwrap(),
        1e-10,
    );
    // Starting from an eigenvector of eigenvalue 0 gives ARPACK a zero
    // Krylov vector: a (recoverable) ARPACK error.
    let zero_start = ArpackOptions::default()
        .with_start(vec![1.0; n])
        .with_which(ArpackWhich::SmallestMagnitude);
    let err = arpack_rssolve(n, path_laplacian_matvec(n), &zero_start, None).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::Arpack);
    // Thread-safe, unlike `arpack_last_error` (a process-wide C variable that
    // the parallel tests of this file overwrite).
    assert_eq!(ArpackError::from_error(&err), Some(ArpackError::ZeroStart));
    assert_eq!(ArpackError::from_error(&Error::invalid("not ARPACK")), None);

    // 1x1 and 2x2 problems are solved exactly by igraph.
    let res = arpack_rssolve(
        1,
        |x: &[f64], y: &mut [f64]| y[0] = 5.0 * x[0],
        &ArpackOptions::default(),
        None,
    )
    .unwrap();
    assert_eq!(res.values, vec![5.0]);
    let two = |x: &[f64], y: &mut [f64]| {
        y[0] = 2.0 * x[0] + x[1];
        y[1] = x[0] + 2.0 * x[1];
    };
    let opts = ArpackOptions::default()
        .with_nev(2)
        .with_which(ArpackWhich::LargestAlgebraic);
    let res = arpack_rssolve(2, two, &opts, None).unwrap();
    assert_vec_close(&res.values, &[3.0, 1.0], 1e-12);
}

#[test]
fn arpack_shift_invert_with_a_sparse_lu() {
    // Interior eigenvalues: the ones closest to sigma = 1 of the path Laplacian,
    // with a closure applying (L - sigma I)^-1 through a sparse LU.
    let n = 40;
    let sigma = 1.0;
    let l = path_laplacian(n);
    let shifted = l
        .add(&SparseMat::identity(n).unwrap(), 1.0, -sigma)
        .unwrap();
    let lu = shifted.lu(SparseOrdering::MinDegreeSymmetric, 1.0).unwrap();
    let opts = ArpackOptions::default()
        .with_nev(2)
        .with_shift_invert(sigma);
    let res = arpack_rssolve(
        n,
        |x: &[f64], y: &mut [f64]| y.copy_from_slice(&lu.solve(x).unwrap()),
        &opts,
        None,
    )
    .unwrap();
    let mut spectrum = path_laplacian_spectrum(n);
    spectrum.sort_by(|a, b| (a - sigma).abs().total_cmp(&(b - sigma).abs()));
    assert_vec_close(
        &sorted(res.values.clone()),
        &sorted(spectrum[..2].to_vec()),
        1e-8,
    );

    // The same with igraph's own sparse shift-and-invert driver, LU and QR.
    for method in [SparseSolveMethod::Lu, SparseSolveMethod::Qr] {
        let res = l.arpack_rssolve(&opts, method).unwrap();
        assert_vec_close(
            &sorted(res.values.clone()),
            &sorted(spectrum[..2].to_vec()),
            1e-8,
        );
    }
    // Regular mode on the sparse matrix.
    let top = l
        .arpack_rssolve(&ArpackOptions::default().with_nev(1), SparseSolveMethod::Lu)
        .unwrap();
    assert_close(
        top.values[0],
        *path_laplacian_spectrum(n).last().unwrap(),
        1e-8,
    );
}

#[test]
fn arpack_nonsymmetric_directed_cycle() {
    rng::seed(7).unwrap();
    let n = 9;
    // Permutation matrix of a directed 9-cycle: eigenvalues exp(2 pi i k / 9).
    let shift = |x: &[f64], y: &mut [f64]| {
        for i in 0..n {
            y[i] = x[(i + 1) % n];
        }
    };
    let opts = ArpackOptions::default()
        .with_nev(3)
        .with_which(ArpackWhich::LargestReal);
    let res = arpack_rnsolve(n, shift, &opts, None).unwrap();
    assert_eq!(res.values.len(), res.vectors.len());
    assert_close(res.values[0].re(), 1.0, 1e-8);
    assert_close(res.values[0].im(), 0.0, 1e-8);
    for (k, v) in res.values.iter().enumerate().skip(1) {
        assert_close(v.re(), (2.0 * PI / n as f64).cos(), 1e-8);
        assert_close(v.im().abs(), (2.0 * PI / n as f64).sin(), 1e-8);
        // Complex conjugate pairs.
        if k == 2 {
            assert_close(v.im(), -res.values[1].im(), 1e-12);
        }
    }
    // Check P v = lambda v for the complex eigenvectors.
    for (lambda, v) in res.values.iter().zip(&res.vectors) {
        for i in 0..n {
            let pv = v[(i + 1) % n];
            let lv = (
                lambda.re() * v[i].re() - lambda.im() * v[i].im(),
                lambda.re() * v[i].im() + lambda.im() * v[i].re(),
            );
            assert_close(pv.re(), lv.0, 1e-7);
            assert_close(pv.im(), lv.1, 1e-7);
        }
    }

    // With a non-symmetric storage.
    let mut storage = ArpackStorage::new(n, n, false).unwrap();
    let res = arpack_rnsolve(n, shift, &opts, Some(&mut storage)).unwrap();
    assert_close(res.values[0].re(), 1.0, 1e-8);
    let mut sym = ArpackStorage::new(n, n, true).unwrap();
    assert!(arpack_rnsolve(n, shift, &opts, Some(&mut sym)).is_err());
}

#[test]
fn arpack_small_nonsymmetric_and_sparse() {
    // 2x2 rotation-like matrix with complex eigenvalues 1 ± 2i.
    let m = |x: &[f64], y: &mut [f64]| {
        y[0] = x[0] - 2.0 * x[1];
        y[1] = 2.0 * x[0] + x[1];
    };
    let res = arpack_rnsolve(2, m, &ArpackOptions::default().with_nev(1), None).unwrap();
    assert_close(res.values[0].re(), 1.0, 1e-12);
    assert_close(res.values[0].im().abs(), 2.0, 1e-12);

    // Sparse non-symmetric: an upper triangular matrix has its diagonal as
    // spectrum.
    let t = SparseMat::from_triplets(
        6,
        6,
        &[
            (0, 0, 6.0),
            (1, 1, 5.0),
            (2, 2, 4.0),
            (3, 3, 3.0),
            (4, 4, 2.0),
            (5, 5, 1.0),
            (0, 3, 1.0),
            (2, 5, 2.0),
            (1, 4, -1.0),
        ],
    )
    .unwrap();
    let opts = ArpackOptions::default()
        .with_nev(2)
        .with_which(ArpackWhich::LargestMagnitude);
    let res = t.arpack_rnsolve(&opts).unwrap();
    assert_close(res.values[0].re(), 6.0, 1e-8);
    assert_close(res.values[1].re(), 5.0, 1e-8);
    assert!(
        t.arpack_rnsolve(&opts.clone().with_shift_invert(1.0))
            .is_err()
    );
}

#[test]
fn arpack_invalid_options_and_errors() {
    let n = 10;
    let f = path_laplacian_matvec(n);
    let err = |o: ArpackOptions| {
        arpack_rssolve(n, path_laplacian_matvec(n), &o, None)
            .unwrap_err()
            .kind()
    };
    assert_eq!(
        err(ArpackOptions::default().with_nev(0)),
        ErrorKind::InvalidValue
    );
    assert_eq!(
        err(ArpackOptions::default().with_which(ArpackWhich::LargestReal)),
        ErrorKind::InvalidValue
    );
    assert_eq!(
        err(ArpackOptions::default().with_start(vec![1.0; 3])),
        ErrorKind::InvalidValue
    );
    assert_eq!(
        err(ArpackOptions::default().with_ncv(11)),
        ErrorKind::InvalidValue
    );
    assert_eq!(
        err(ArpackOptions::default().with_mxiter(0)),
        ErrorKind::InvalidValue
    );
    assert!(arpack_rssolve(0, f, &ArpackOptions::default(), None).is_err());
    assert!(
        arpack_rnsolve(
            n,
            path_laplacian_matvec(n),
            &ArpackOptions::default().with_which(ArpackWhich::BothEnds),
            None
        )
        .is_err()
    );

    // ARPACK itself refuses nev >= n; the error condition is recognized from
    // the returned error.
    let e = arpack_rssolve(
        n,
        path_laplacian_matvec(n),
        &ArpackOptions::default().with_nev(n),
        None,
    )
    .unwrap_err();
    assert_eq!(e.kind(), ErrorKind::Arpack);
    let code = ArpackError::from_error(&e).unwrap();
    assert_ne!(code, ArpackError::NoError);
    assert_eq!(arpack_error_to_string(code), e.message());
    assert_eq!(format!("{}", ArpackError::NoError), "No error");

    assert_eq!(ArpackWhich::LargestAlgebraic.code(), "LA");
    assert!(
        ArpackWhich::LargestMagnitude.is_symmetric()
            && ArpackWhich::LargestMagnitude.is_nonsymmetric()
    );
    assert!(!ArpackWhich::LargestImaginary.is_symmetric());
}

#[test]
fn non_finite_values_are_errors_not_aborts() {
    // NaN or infinite values reaching ARPACK's LAPACK routines would abort
    // the whole process: they are reported as errors instead.
    let err = arpack_rssolve(
        10,
        |_: &[f64], y: &mut [f64]| y.fill(f64::NAN),
        &ArpackOptions::default(),
        None,
    )
    .unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidValue);
    assert!(err.message().contains("NaN"));
    let err = arpack_rnsolve(
        10,
        |_: &[f64], y: &mut [f64]| y[3] = f64::INFINITY,
        &ArpackOptions::default(),
        None,
    )
    .unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidValue);
    let bad = dense(&[&[1.0, f64::NAN], &[0.0, 1.0]]);
    assert!(lapack_dsyevr(&bad, &SymmetricRange::All, 0.0).is_err());
    assert!(lapack_dgeev(&bad, false, false).is_err());
    assert!(eigen_matrix(&bad, &EigenWhich::All, EigenAlgorithm::Lapack).is_err());
    assert!(
        eigen_matrix_symmetric(
            &bad,
            &EigenWhich::All,
            EigenAlgorithm::Lapack,
            &ArpackOptions::default()
        )
        .is_err()
    );
    let sbad =
        SparseMat::from_triplets(3, 3, &[(0, 0, f64::NAN), (1, 1, 1.0), (2, 2, 1.0)]).unwrap();
    assert!(
        sbad.arpack_rssolve(&ArpackOptions::default(), SparseSolveMethod::Lu)
            .is_err()
    );
    assert!(
        sbad.eigen_symmetric(
            &EigenWhich::All,
            EigenAlgorithm::Lapack,
            &ArpackOptions::default()
        )
        .is_err()
    );
    let err = eigen_symmetric_fn(
        4,
        |_: &[f64], y: &mut [f64]| y[0] = f64::NAN,
        &EigenWhich::All,
        EigenAlgorithm::Lapack,
        &ArpackOptions::default(),
    )
    .unwrap_err();
    assert!(err.message().contains("NaN"));
    let nan_tol = ArpackOptions::default().with_tol(f64::NAN);
    assert!(arpack_rssolve(10, path_laplacian_matvec(10), &nan_tol, None).is_err());
    // Normalized Laplacians of graphs with isolated vertices would divide by zero.
    let g = Graph::from_edges(&[(0, 1), (1, 2)], 4, false).unwrap();
    let o = ArpackOptions::default();
    for kind in [LaplacianEmbeddingType::DAD, LaplacianEmbeddingType::IDAD] {
        assert!(
            g.laplacian_spectral_embedding(
                1,
                None,
                EmbeddingWhich::LargestAlgebraic,
                kind,
                true,
                &o
            )
            .is_err()
        );
    }
    assert!(
        g.adjacency_spectral_embedding(
            1,
            Some(&[1.0, f64::NAN]),
            EmbeddingWhich::LargestAlgebraic,
            true,
            None,
            &o
        )
        .is_err()
    );
    // Shift-and-invert exactly at an eigenvalue: A - sigma I is singular.
    let l = path_laplacian(5);
    let r = l.arpack_rssolve(
        &ArpackOptions::default().with_shift_invert(0.0),
        SparseSolveMethod::Lu,
    );
    assert!(r.is_err());
}

#[test]
fn arpack_panicking_callback_is_resumed() {
    let r = std::panic::catch_unwind(|| {
        let _ = arpack_rssolve(
            10,
            |_: &[f64], _: &mut [f64]| panic!("matvec failed"),
            &ArpackOptions::default(),
            None,
        );
    });
    let payload = r.unwrap_err();
    assert_eq!(payload.downcast_ref::<&str>(), Some(&"matvec failed"));
    // igraph still works on this thread.
    let res = arpack_rssolve(
        10,
        path_laplacian_matvec(10),
        &ArpackOptions::default(),
        None,
    )
    .unwrap();
    assert!(res.values[0] > 3.9);
}

#[test]
fn arpack_unpack_complex_format() {
    // Eigenvalues: 2 (real), 1 ± 3i (pair); packed vectors: 1 + 2 columns.
    let values = dense(&[&[2.0, 0.0], &[1.0, 3.0], &[1.0, -3.0]]);
    let vectors = dense(&[&[1.0, 0.5, 0.25], &[0.0, 1.0, -1.0]]);
    let (v, vals) = arpack_unpack_complex(&vectors, &values, 3).unwrap();
    assert_eq!(vals.shape(), (3, 2));
    assert_eq!(v.shape(), (2, 6));
    // Real eigenvector: imaginary part zero.
    assert_eq!(v.column(0), &[1.0, 0.0]);
    assert_eq!(v.column(1), &[0.0, 0.0]);
    // The pair: v and its conjugate.
    assert_eq!(v.column(2), &[0.5, 1.0]);
    assert_eq!(v.column(3), &[0.25, -1.0]);
    assert_eq!(v.column(4), &[0.5, 1.0]);
    assert_eq!(v.column(5), &[-0.25, 1.0]);
    // Keep only the first eigenvalue.
    let (v1, vals1) = arpack_unpack_complex(&vectors, &values, 1).unwrap();
    assert_eq!((v1.shape(), vals1.shape()), ((2, 2), (1, 2)));
    // Malformed inputs.
    assert!(arpack_unpack_complex(&vectors, &values, 4).is_err());
    assert!(arpack_unpack_complex(&vectors, &dense(&[&[1.0]]), 1).is_err());
    assert!(arpack_unpack_complex(&dense(&[&[1.0, 2.0]]), &values, 3).is_err());
}

// ---------------------------------------------------------------------------
// Eigen front-end
// ---------------------------------------------------------------------------

#[test]
fn eigen_symmetric_lapack_and_arpack_agree() {
    // A = Q diag(lambda) Q' with a planted, well separated spectrum (the
    // orthogonal Q comes from the eigenvectors of a path Laplacian): igraph
    // runs ARPACK with few Lanczos vectors here, so the extremes must stand out.
    let n = 30;
    let mut planted: Vec<f64> = (0..n - 4)
        .map(|i| -3.0 + 6.0 * i as f64 / (n - 5) as f64)
        .collect();
    planted.extend([-18.0, -12.0, 14.0, 20.0]);
    let q = lapack_dsyevr(
        &path_laplacian(n).to_dense().unwrap(),
        &SymmetricRange::All,
        1e-14,
    )
    .unwrap()
    .vectors;
    let mut ql = q.clone();
    for (j, &l) in planted.iter().enumerate() {
        for i in 0..n {
            ql[(i, j)] *= l;
        }
    }
    let a = blas_dgemm(false, true, 1.0, &ql, &q, 0.0, None).unwrap();
    let all = eigen_matrix_symmetric(
        &a,
        &EigenWhich::All,
        EigenAlgorithm::Lapack,
        &ArpackOptions::default(),
    )
    .unwrap();
    assert_eq!(all.values.len(), n);
    let spectrum = sorted(all.values.clone());
    assert_vec_close(&spectrum, &sorted(planted), 1e-9);
    // Trace = sum of eigenvalues.
    let trace: f64 = (0..n).map(|i| a[(i, i)]).sum();
    assert_close(spectrum.iter().sum(), trace, 1e-9);

    let by_mag = {
        let mut s = spectrum.clone();
        s.sort_by(|x, y| y.abs().total_cmp(&x.abs()));
        s
    };
    for algorithm in [
        EigenAlgorithm::Lapack,
        EigenAlgorithm::Arpack,
        EigenAlgorithm::Auto,
    ] {
        let opts = ArpackOptions::default();
        let la =
            eigen_matrix_symmetric(&a, &EigenWhich::LargestAlgebraic(3), algorithm, &opts).unwrap();
        assert_vec_close(&sorted(la.values), &spectrum[n - 3..], 1e-8);
        let sa = eigen_matrix_symmetric(&a, &EigenWhich::SmallestAlgebraic(2), algorithm, &opts)
            .unwrap();
        assert_vec_close(&sorted(sa.values), &spectrum[..2], 1e-8);
        let lm =
            eigen_matrix_symmetric(&a, &EigenWhich::LargestMagnitude(2), algorithm, &opts).unwrap();
        assert_vec_close(&sorted(lm.values), &sorted(by_mag[..2].to_vec()), 1e-8);
        let be = eigen_matrix_symmetric(&a, &EigenWhich::BothEnds(4), algorithm, &opts).unwrap();
        let expected = sorted(vec![
            spectrum[0],
            spectrum[1],
            spectrum[n - 2],
            spectrum[n - 1],
        ]);
        assert_vec_close(&sorted(be.values), &expected, 1e-8);
    }
    let sm = eigen_matrix_symmetric(
        &a,
        &EigenWhich::SmallestMagnitude(2),
        EigenAlgorithm::Lapack,
        &ArpackOptions::default(),
    )
    .unwrap();
    assert_vec_close(&sorted(sm.values), &sorted(by_mag[n - 2..].to_vec()), 1e-8);

    let sel = eigen_matrix_symmetric(
        &a,
        &EigenWhich::Select(2..5),
        EigenAlgorithm::Lapack,
        &ArpackOptions::default(),
    )
    .unwrap();
    assert_vec_close(&sel.values, &spectrum[2..5], 1e-9);
    let (lo, hi) = (spectrum[3] - 1e-6, spectrum[6] + 1e-6);
    let iv = eigen_matrix_symmetric(
        &a,
        &EigenWhich::Interval { low: lo, high: hi },
        EigenAlgorithm::Lapack,
        &ArpackOptions::default(),
    )
    .unwrap();
    assert_vec_close(&iv.values, &spectrum[3..7], 1e-9);
    assert_eq!(iv.vectors.shape(), (n, 4));

    // Unsupported or invalid choices.
    let opts = ArpackOptions::default();
    assert_eq!(
        eigen_matrix_symmetric(&a, &EigenWhich::Select(0..2), EigenAlgorithm::Arpack, &opts)
            .unwrap_err()
            .kind(),
        ErrorKind::Unimplemented
    );
    assert!(
        eigen_matrix_symmetric(
            &a,
            &EigenWhich::LargestAlgebraic(0),
            EigenAlgorithm::Lapack,
            &opts
        )
        .is_err()
    );
    assert!(
        eigen_matrix_symmetric(
            &a,
            &EigenWhich::LargestAlgebraic(n + 1),
            EigenAlgorithm::Lapack,
            &opts
        )
        .is_err()
    );
    assert!(
        eigen_matrix_symmetric(&a, &EigenWhich::Select(3..3), EigenAlgorithm::Lapack, &opts)
            .is_err()
    );
    assert!(
        eigen_matrix_symmetric(
            &a,
            &EigenWhich::Interval {
                low: 1.0,
                high: 0.0
            },
            EigenAlgorithm::Lapack,
            &opts
        )
        .is_err()
    );
    assert!(
        eigen_matrix_symmetric(
            &a,
            &EigenWhich::LargestReal(1),
            EigenAlgorithm::Lapack,
            &opts
        )
        .is_err()
    );
    assert!(
        eigen_matrix_symmetric(
            &Matrix::zeros(2, 3),
            &EigenWhich::All,
            EigenAlgorithm::Lapack,
            &opts
        )
        .is_err()
    );
    assert!(
        eigen_matrix_symmetric(
            &Matrix::new(),
            &EigenWhich::All,
            EigenAlgorithm::Lapack,
            &opts
        )
        .is_err()
    );
}

#[test]
fn eigen_matrix_symmetric_matches_igraph_unit_test() {
    // tests/unit/igraph_eigen_matrix_symmetric.c and .out: a 10 x 10
    // symmetric matrix with entries drawn from igraph's default RNG seeded
    // with 42 * 42 (the per-thread RNG of this crate is the same generator).
    rng::seed(42 * 42).unwrap();
    let dim = 10;
    let mut a = Matrix::zeros(dim, dim);
    for i in 0..dim {
        for j in i..dim {
            let x = rng::integer(1, 10) as f64;
            a[(i, j)] = x;
            a[(j, i)] = x;
        }
    }
    let run = |which: EigenWhich| {
        let e = eigen_matrix_symmetric(
            &a,
            &which,
            EigenAlgorithm::Lapack,
            &ArpackOptions::default(),
        )
        .unwrap();
        // Each column is an eigenvector of its eigenvalue.
        for (k, lambda) in e.values.iter().enumerate() {
            let v = e.vectors.column(k);
            let av = blas_dgemv(false, 1.0, &a, v, 0.0, &vec![0.0; dim]).unwrap();
            for i in 0..dim {
                assert_close(av[i], lambda * v[i], 1e-9);
            }
        }
        e.values
    };
    let lm5 = [54.2836, -14.0379, 11.141, -10.6996, -8.73724];
    assert_vec_close(&run(EigenWhich::LargestMagnitude(5)), &lm5, 1e-4);
    let lm8 = [
        54.2836, -14.0379, 11.141, -10.6996, -8.73724, 8.62656, 7.57286, 4.18242,
    ];
    assert_vec_close(&run(EigenWhich::LargestMagnitude(8)), &lm8, 1e-4);
    let be5 = [54.2836, -14.0379, 11.141, -10.6996, 8.62656];
    assert_vec_close(&run(EigenWhich::BothEnds(5)), &be5, 1e-4);
    let sm5 = [-1.76375, 2.43199, 4.18242, 7.57286, 8.62656];
    // In increasing magnitude, as igraph prints them.
    assert_vec_close(&run(EigenWhich::SmallestMagnitude(5)), &sm5, 1e-4);
}

#[test]
fn eigen_symmetric_ordering_and_smallest_magnitude() {
    // The path Laplacian plus the identity is positive definite, so its
    // eigenvalue of smallest magnitude is the smallest one: igraph's own
    // LAPACK "smallest magnitude" code reads outside its arrays in this
    // case, the wrapper selects the eigenpairs itself.
    let n = 8;
    let shifted = (&path_laplacian(n) + &SparseMat::identity(n).unwrap()).unwrap();
    let dense = shifted.to_dense().unwrap();
    let spectrum: Vec<f64> = path_laplacian_spectrum(n).iter().map(|x| x + 1.0).collect();
    let opts = ArpackOptions::default();
    for algorithm in [EigenAlgorithm::Lapack, EigenAlgorithm::Auto] {
        let sm =
            eigen_matrix_symmetric(&dense, &EigenWhich::SmallestMagnitude(3), algorithm, &opts)
                .unwrap();
        assert_vec_close(&sm.values, &spectrum[..3], 1e-10);
        assert_eq!(sm.vectors.shape(), (n, 3));
        // Each column is a unit eigenvector of its eigenvalue.
        for (k, &lambda) in sm.values.iter().enumerate() {
            let v = sm.vectors.column(k);
            let av = shifted.mul_vec(v).unwrap();
            for i in 0..n {
                assert_close(av[i], lambda * v[i], 1e-9);
            }
            assert_close(blas_dnrm2(v).unwrap(), 1.0, 1e-10);
        }
        // Also from a sparse matrix and from a closure.
        let s = shifted
            .eigen_symmetric(&EigenWhich::SmallestMagnitude(2), algorithm, &opts)
            .unwrap();
        assert_vec_close(&s.values, &spectrum[..2], 1e-10);
        let mut matvec = path_laplacian_matvec(n);
        let f = eigen_symmetric_fn(
            n,
            |x: &[f64], y: &mut [f64]| {
                matvec(x, y);
                for i in 0..n {
                    y[i] += x[i];
                }
            },
            &EigenWhich::SmallestMagnitude(1),
            algorithm,
            &opts,
        )
        .unwrap();
        assert_vec_close(&f.values, &spectrum[..1], 1e-10);
    }
    // A negative definite matrix: the smallest magnitude is the largest value.
    let mut neg = dense.clone();
    neg.as_mut_slice().iter_mut().for_each(|x| *x = -*x);
    let sm = eigen_matrix_symmetric(
        &neg,
        &EigenWhich::SmallestMagnitude(2),
        EigenAlgorithm::Lapack,
        &opts,
    )
    .unwrap();
    assert_vec_close(&sm.values, &[-spectrum[0], -spectrum[1]], 1e-10);

    // The order of the eigenvalues is the same whichever algorithm runs.
    let desc: Vec<f64> = spectrum.iter().rev().copied().collect();
    for algorithm in [EigenAlgorithm::Lapack, EigenAlgorithm::Arpack] {
        let la = eigen_matrix_symmetric(&dense, &EigenWhich::LargestAlgebraic(3), algorithm, &opts)
            .unwrap();
        assert_vec_close(&la.values, &desc[..3], 1e-8);
        let lm = eigen_matrix_symmetric(&dense, &EigenWhich::LargestMagnitude(2), algorithm, &opts)
            .unwrap();
        assert_vec_close(&lm.values, &desc[..2], 1e-8);
        let sa =
            eigen_matrix_symmetric(&dense, &EigenWhich::SmallestAlgebraic(3), algorithm, &opts)
                .unwrap();
        assert_vec_close(&sa.values, &spectrum[..3], 1e-8);
        // Eigenvectors follow their eigenvalues.
        let v = la.vectors.column(0);
        let av = shifted.mul_vec(v).unwrap();
        for i in 0..n {
            assert_close(av[i], desc[0] * v[i], 1e-7);
        }
    }
    let all =
        eigen_matrix_symmetric(&dense, &EigenWhich::All, EigenAlgorithm::Lapack, &opts).unwrap();
    assert_vec_close(&all.values, &spectrum, 1e-10);
    // BothEnds alternates largest, smallest, second largest, ...
    let be = eigen_matrix_symmetric(
        &dense,
        &EigenWhich::BothEnds(3),
        EigenAlgorithm::Lapack,
        &opts,
    )
    .unwrap();
    assert_vec_close(&be.values, &[desc[0], spectrum[0], desc[1]], 1e-10);
}

#[test]
fn eigen_symmetric_from_closure_and_sparse() {
    let n = 25;
    let spectrum = path_laplacian_spectrum(n);
    for algorithm in [EigenAlgorithm::Lapack, EigenAlgorithm::Arpack] {
        let e = eigen_symmetric_fn(
            n,
            path_laplacian_matvec(n),
            &EigenWhich::LargestAlgebraic(2),
            algorithm,
            &ArpackOptions::default(),
        )
        .unwrap();
        assert_vec_close(&sorted(e.values), &spectrum[n - 2..], 1e-8);
        let s = path_laplacian(n)
            .eigen_symmetric(
                &EigenWhich::SmallestAlgebraic(2),
                algorithm,
                &ArpackOptions::default(),
            )
            .unwrap();
        assert_vec_close(&sorted(s.values), &spectrum[..2], 1e-8);
    }
    // A triplet matrix works too (compressed on the fly).
    let t = SparseMat::from_triplets(n, n, &laplacian_triplets(n, &path_edges(n))).unwrap();
    let all = t
        .eigen_symmetric(
            &EigenWhich::All,
            EigenAlgorithm::Lapack,
            &ArpackOptions::default(),
        )
        .unwrap();
    assert_vec_close(&sorted(all.values), &spectrum, 1e-9);
}

#[test]
fn eigen_general_matrices() {
    // Companion matrix of (x - 1)(x - 2)(x - 3) = x^3 - 6x^2 + 11x - 6.
    let c = dense(&[&[6.0, -11.0, 6.0], &[1.0, 0.0, 0.0], &[0.0, 1.0, 0.0]]);
    let e = eigen_matrix(&c, &EigenWhich::All, EigenAlgorithm::Lapack).unwrap();
    let re = sorted(e.values.iter().map(|v| v.re()).collect());
    assert_vec_close(&re, &[1.0, 2.0, 3.0], 1e-10);
    assert!(e.values.iter().all(|v| v.im().abs() < 1e-12));
    let lm = eigen_matrix(&c, &EigenWhich::LargestMagnitude(1), EigenAlgorithm::Lapack).unwrap();
    assert_close(lm.values[0].re(), 3.0, 1e-10);
    // A v = 3 v.
    let v: Vec<f64> = lm.vectors[0].iter().map(|z| z.re()).collect();
    let av = blas_dgemv(false, 1.0, &c, &v, 0.0, &[0.0; 3]).unwrap();
    assert_vec_close(&av, &v.iter().map(|x| 3.0 * x).collect::<Vec<_>>(), 1e-10);
    let sr = eigen_matrix(&c, &EigenWhich::SmallestReal(1), EigenAlgorithm::Lapack).unwrap();
    assert_close(sr.values[0].re(), 1.0, 1e-10);
    let sel = eigen_matrix(&c, &EigenWhich::Select(0..2), EigenAlgorithm::Lapack).unwrap();
    assert_eq!(sel.values.len(), 2);

    // Rotation by 90 degrees plus a real eigenvalue 2.
    let r = dense(&[&[0.0, -1.0, 0.0], &[1.0, 0.0, 0.0], &[0.0, 0.0, 2.0]]);
    let li = eigen_matrix(&r, &EigenWhich::LargestImaginary(1), EigenAlgorithm::Lapack).unwrap();
    assert_close(li.values[0].im(), 1.0, 1e-12);
    let si = eigen_matrix(
        &r,
        &EigenWhich::SmallestImaginary(1),
        EigenAlgorithm::Lapack,
    )
    .unwrap();
    assert_close(si.values[0].im(), -1.0, 1e-12);
    let lr = eigen_matrix(&r, &EigenWhich::LargestReal(1), EigenAlgorithm::Lapack).unwrap();
    assert_close(lr.values[0].re(), 2.0, 1e-12);
    // The eigenvector of i: (1, -i, 0)/sqrt(2) up to a phase; check R v = i v.
    let v = &li.vectors[0];
    let rv0 = (-v[1].re(), -v[1].im());
    assert_close(rv0.0, -v[0].im(), 1e-12);
    assert_close(rv0.1, v[0].re(), 1e-12);
    let sm = eigen_matrix(
        &r,
        &EigenWhich::SmallestMagnitude(2),
        EigenAlgorithm::Lapack,
    )
    .unwrap();
    assert!(
        sm.values
            .iter()
            .all(|v| (v.re().hypot(v.im()) - 1.0).abs() < 1e-12)
    );

    // Sparse input.
    let s = SparseMat::from_dense(&c, 0.0).unwrap();
    let se = s.eigen(&EigenWhich::All, EigenAlgorithm::Lapack).unwrap();
    assert_vec_close(
        &sorted(se.values.iter().map(|v| v.re()).collect()),
        &[1.0, 2.0, 3.0],
        1e-10,
    );

    // Only LAPACK is implemented; BothEnds is not for general matrices.
    assert_eq!(
        eigen_matrix(&c, &EigenWhich::All, EigenAlgorithm::Arpack)
            .unwrap_err()
            .kind(),
        ErrorKind::Unimplemented
    );
    assert!(eigen_matrix(&c, &EigenWhich::BothEnds(2), EigenAlgorithm::Lapack).is_err());
    assert!(
        eigen_matrix(
            &Matrix::zeros(2, 3),
            &EigenWhich::All,
            EigenAlgorithm::Lapack
        )
        .is_err()
    );
}

#[test]
fn eigen_adjacency_petersen_spectrum() {
    rng::seed(1).unwrap();
    let g = petersen();
    let opts = ArpackOptions::default();
    let top = g
        .eigen_adjacency(
            &EigenWhich::LargestAlgebraic(1),
            EigenAlgorithm::Auto,
            &opts,
        )
        .unwrap();
    assert_close(top.values[0], 3.0, 1e-10);
    // The Perron vector of a regular graph is constant.
    let v = top.vectors.column(0);
    assert!(
        v.iter()
            .all(|x| (x.abs() - 1.0 / 10f64.sqrt()).abs() < 1e-8)
    );
    let bottom = g
        .eigen_adjacency(
            &EigenWhich::SmallestAlgebraic(1),
            EigenAlgorithm::Arpack,
            &opts,
        )
        .unwrap();
    assert_close(bottom.values[0], -2.0, 1e-8);
    let lm = g
        .eigen_adjacency(
            &EigenWhich::LargestMagnitude(1),
            EigenAlgorithm::Arpack,
            &opts,
        )
        .unwrap();
    assert_close(lm.values[0], 3.0, 1e-8);

    // Cross-check with the dense matrix and LAPACK: 3, 1 x5, -2 x4.
    let a = g
        .get_adjacency(GetAdjacency::Both, None, Loops::Twice)
        .unwrap();
    let all = lapack_dsyevr(&a, &SymmetricRange::All, 1e-12).unwrap();
    let expected = [-2.0, -2.0, -2.0, -2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 3.0];
    assert_vec_close(&all.values, &expected, 1e-10);
    // Tr(A^2) = 2|E| and Tr(A^3) = 6 * #triangles = 0 (girth 5).
    assert_close(all.values.iter().map(|x| x * x).sum(), 30.0, 1e-9);
    assert_close(all.values.iter().map(|x| x * x * x).sum(), 0.0, 1e-9);

    // The Perron vector is the eigenvector centrality (scaled to max 1).
    let ec = g.eigenvector_centrality(NeighborMode::All, None).unwrap();
    assert_close(ec.value, top.values[0], 1e-10);
    let scale = v[0];
    for (x, c) in v.iter().zip(&ec.scores) {
        assert_close(x / scale, *c, 1e-8);
    }

    // Errors: directed graphs, LAPACK, unsupported choices.
    let d = Graph::from_edges(&[(0, 1), (1, 2)], 3, true).unwrap();
    assert_eq!(
        d.eigen_adjacency(
            &EigenWhich::LargestAlgebraic(1),
            EigenAlgorithm::Auto,
            &opts
        )
        .unwrap_err()
        .kind(),
        ErrorKind::Unimplemented
    );
    assert_eq!(
        g.eigen_adjacency(
            &EigenWhich::LargestAlgebraic(1),
            EigenAlgorithm::Lapack,
            &opts
        )
        .unwrap_err()
        .kind(),
        ErrorKind::Unimplemented
    );
    assert!(
        g.eigen_adjacency(&EigenWhich::Select(0..2), EigenAlgorithm::Arpack, &opts)
            .is_err()
    );
    assert!(
        Graph::new(0, false)
            .eigen_adjacency(&EigenWhich::All, EigenAlgorithm::Auto, &opts)
            .is_err()
    );
}

/// Scales a vector so that its entry of largest magnitude is `1`.
fn scale_to_max_one(v: &[f64]) -> Vec<f64> {
    let m = v
        .iter()
        .copied()
        .fold(0.0f64, |a, x| if x.abs() > a.abs() { x } else { a });
    v.iter().map(|x| x / m).collect()
}

#[test]
fn spectral_solvers_agree_with_the_centrality_module() {
    // Eigenvector centrality of the karate club is the Perron vector of its
    // adjacency matrix: ARPACK (eigen_adjacency), dense LAPACK and the
    // centrality module all agree.
    rng::seed(17).unwrap();
    let g = Graph::famous("Zachary").unwrap();
    let top = g
        .eigen_adjacency(
            &EigenWhich::LargestAlgebraic(1),
            EigenAlgorithm::Arpack,
            &ArpackOptions::default(),
        )
        .unwrap();
    let a = g
        .get_adjacency(GetAdjacency::Both, None, Loops::Twice)
        .unwrap();
    let dense = lapack_dsyevr(&a, &SymmetricRange::Select(33..34), 1e-12).unwrap();
    let ec = g.eigenvector_centrality(NeighborMode::All, None).unwrap();
    // The leading adjacency eigenvalue of the karate club.
    assert_close(ec.value, 6.725697727, 1e-8);
    assert_close(top.values[0], ec.value, 1e-10);
    assert_close(dense.values[0], ec.value, 1e-10);
    assert_vec_close(&scale_to_max_one(top.vectors.column(0)), &ec.scores, 1e-8);
    assert_vec_close(&scale_to_max_one(dense.vectors.column(0)), &ec.scores, 1e-8);
    // Vertex 33 (the administrator) is the most central one.
    assert_eq!(ec.scores[33], 1.0);

    // Kleinberg's hubs and authorities are the principal eigenvectors of
    // A A^T and A^T A: form them with BLAS, diagonalize them with LAPACK.
    let web = Graph::from_edges(
        &[
            (0, 2),
            (0, 3),
            (1, 2),
            (1, 3),
            (1, 4),
            (4, 3),
            (5, 3),
            (2, 3),
        ],
        6,
        true,
    )
    .unwrap();
    let a = web
        .get_adjacency(GetAdjacency::Both, None, Loops::Twice)
        .unwrap();
    let aat = blas_dgemm(false, true, 1.0, &a, &a, 0.0, None).unwrap();
    let ata = blas_dgemm(true, false, 1.0, &a, &a, 0.0, None).unwrap();
    let hubs = lapack_dsyevr(&aat, &SymmetricRange::Select(5..6), 1e-12).unwrap();
    let auths = lapack_dsyevr(&ata, &SymmetricRange::Select(5..6), 1e-12).unwrap();
    // A A^T and A^T A share their non-zero spectrum.
    assert_close(hubs.values[0], auths.values[0], 1e-10);
    let hits = web.hub_and_authority_scores(None).unwrap();
    assert_close(hits.value, hubs.values[0], 1e-10);
    assert_vec_close(&scale_to_max_one(hubs.vectors.column(0)), &hits.hubs, 1e-8);
    assert_vec_close(
        &scale_to_max_one(auths.vectors.column(0)),
        &hits.authorities,
        1e-8,
    );
}

#[test]
fn seeded_arpack_runs_are_reproducible_in_parallel_threads() {
    // Without a start vector ARPACK starts from a random vector drawn from
    // the calling thread's igraph RNG: seeding makes it reproducible, and
    // every thread has its own RNG, so parallel runs do not interfere.
    fn first_product_input(seed: u64) -> (Vec<f64>, Vec<f64>) {
        rng::seed(seed).unwrap();
        let n = 30;
        let mut first: Option<Vec<f64>> = None;
        let mut matvec = path_laplacian_matvec(n);
        let opts = ArpackOptions::default()
            .with_nev(2)
            .with_which(ArpackWhich::LargestAlgebraic);
        let res = arpack_rssolve(
            n,
            |x: &[f64], y: &mut [f64]| {
                first.get_or_insert_with(|| x.to_vec());
                matvec(x, y)
            },
            &opts,
            None,
        )
        .unwrap();
        (first.unwrap(), res.values)
    }
    let reference = first_product_input(42);
    let handles: Vec<_> = (0..4)
        .map(|_| std::thread::spawn(|| first_product_input(42)))
        .collect();
    for h in handles {
        let (start, values) = h.join().unwrap();
        assert_eq!(start, reference.0, "same seed, same random start");
        assert_vec_close(&values, &reference.1, 1e-10);
    }
    // A different seed gives a different start vector, but the same answer.
    let (other_start, other_values) = first_product_input(43);
    assert_ne!(other_start, reference.0);
    assert_vec_close(&other_values, &reference.1, 1e-10);
    let spectrum = path_laplacian_spectrum(30);
    assert_vec_close(&sorted(reference.1.clone()), &spectrum[28..], 1e-8);
}

// ---------------------------------------------------------------------------
// LAPACK
// ---------------------------------------------------------------------------

#[test]
fn lapack_linear_systems() {
    let n = 8;
    let a = random_matrix(n, n, 11);
    let x = random_matrix(n, 3, 12);
    let b = matmul(&a, &x);
    let res = lapack_dgesv(&a, &b).unwrap();
    assert_matrix_close(&res.solution, &x, 1e-10);
    assert!(!res.factors.is_singular());
    assert_eq!(res.factors.ipiv.len(), n);

    // Factorize once, solve A X = B and A' X = B.
    let lu = lapack_dgetrf(&a).unwrap();
    assert_eq!(lu.info, 0);
    assert!(lu.ipiv.iter().all(|&p| p >= 1 && p <= n as i64));
    assert_matrix_close(&lu.solve(false, &b).unwrap(), &x, 1e-10);
    let bt = matmul(&a.transposed(), &x);
    assert_matrix_close(
        &lapack_dgetrs(true, &lu.lu, &lu.ipiv, &bt).unwrap(),
        &x,
        1e-10,
    );

    // P L U reconstructs A: check the determinant through the diagonal of U.
    let det_u: f64 = (0..n).map(|i| lu.lu[(i, i)]).product();
    let swaps = lu
        .ipiv
        .iter()
        .enumerate()
        .filter(|&(i, &p)| p as usize != i + 1)
        .count();
    let det = if swaps % 2 == 0 { det_u } else { -det_u };
    let all = lapack_dgeev(&a, false, false).unwrap();
    // det = product of the eigenvalues (complex pairs multiply to |z|^2).
    let mut prod = (1.0, 0.0);
    for v in all.values() {
        prod = (
            prod.0 * v.re() - prod.1 * v.im(),
            prod.0 * v.im() + prod.1 * v.re(),
        );
    }
    assert_close(prod.0, det, 1e-9);

    // Single right hand side helper.
    let xs = solve(&dense(&[&[3.0, 2.0], &[1.0, 2.0]]), &[5.0, 3.0]).unwrap();
    assert_vec_close(&xs, &[1.0, 1.0], 1e-12);

    // Singular systems and bad shapes.
    let sing = dense(&[&[1.0, 2.0], &[2.0, 4.0]]);
    assert_eq!(
        lapack_dgesv(&sing, &dense(&[&[1.0], &[1.0]]))
            .unwrap_err()
            .kind(),
        ErrorKind::Failure
    );
    assert!(lapack_dgetrf(&sing).unwrap().is_singular());
    assert!(lapack_dgesv(&Matrix::zeros(2, 3), &Matrix::zeros(2, 1)).is_err());
    assert!(lapack_dgesv(&a, &Matrix::zeros(n + 1, 1)).is_err());
    assert!(lapack_dgetrs(false, &lu.lu, &[0; 8], &b).is_err());
    assert!(lapack_dgetrs(false, &lu.lu, &lu.ipiv[..3], &b).is_err());

    // Rectangular LU is allowed.
    let rect = lapack_dgetrf(&random_matrix(3, 5, 13)).unwrap();
    assert_eq!(rect.ipiv.len(), 3);
}

#[test]
fn lapack_symmetric_eigenproblems() {
    // The igraph example (examples/simple/igraph_lapack_dsyevr.out).
    let a = dense(&[&[2.0, -1.0], &[-1.0, 3.0]]);
    let low = lapack_dsyevr(&a, &SymmetricRange::Select(0..1), 1e-10).unwrap();
    assert_close(low.values[0], 1.38197, 1e-5);
    let v = low.vectors.column(0);
    assert_close(v[0].abs(), 0.850651, 1e-6);
    assert_close(v[1].abs(), 0.525731, 1e-6);
    let high = lapack_dsyevr(
        &a,
        &SymmetricRange::Interval {
            low: 3.0,
            high: 4.0,
        },
        1e-10,
    )
    .unwrap();
    assert_close(high.values[0], 3.61803, 1e-5);

    // Dense path Laplacian: all eigenvalues, orthonormal eigenvectors.
    let n = 12;
    let l = path_laplacian(n).to_dense().unwrap();
    let all = lapack_dsyevr(&l, &SymmetricRange::All, 1e-12).unwrap();
    assert_vec_close(&all.values, &path_laplacian_spectrum(n), 1e-10);
    let vtv = blas_dgemm(true, false, 1.0, &all.vectors, &all.vectors, 0.0, None).unwrap();
    for i in 0..n {
        for j in 0..n {
            assert_close(vtv[(i, j)], if i == j { 1.0 } else { 0.0 }, 1e-10);
        }
    }
    // An empty interval gives no eigenvalues.
    let none = lapack_dsyevr(
        &l,
        &SymmetricRange::Interval {
            low: 10.0,
            high: 11.0,
        },
        1e-12,
    )
    .unwrap();
    assert!(none.values.is_empty());

    // Invalid ranges are rejected before reaching LAPACK (which would abort).
    assert!(lapack_dsyevr(&l, &SymmetricRange::Select(5..20), 1e-12).is_err());
    assert!(lapack_dsyevr(&l, &SymmetricRange::Select(3..3), 1e-12).is_err());
    assert!(
        lapack_dsyevr(
            &l,
            &SymmetricRange::Interval {
                low: 1.0,
                high: 1.0
            },
            1e-12
        )
        .is_err()
    );
    assert!(lapack_dsyevr(&Matrix::new(), &SymmetricRange::All, 1e-12).is_err());
    assert!(lapack_dsyevr(&Matrix::zeros(2, 3), &SymmetricRange::All, 1e-12).is_err());
}

#[test]
fn lapack_general_eigenproblems() {
    // igraph's example: [[1, 1], [-1, 1]] has eigenvalues 1 ± i and
    // eigenvectors (0.707107, ±0.707107 i).
    let a = dense(&[&[1.0, 1.0], &[-1.0, 1.0]]);
    let e = lapack_dgeev(&a, true, true).unwrap();
    assert_vec_close(&e.values_real, &[1.0, 1.0], 1e-12);
    assert_vec_close(&e.values_imag, &[1.0, -1.0], 1e-12);
    let vr = e.vectors_right.as_ref().unwrap();
    assert_close(vr[(0, 0)], std::f64::consts::FRAC_1_SQRT_2, 1e-6);
    assert_close(vr[(1, 1)], std::f64::consts::FRAC_1_SQRT_2, 1e-6);
    let right = e.right_eigenvectors().unwrap();
    assert_eq!(right.len(), 2);
    assert_close(right[1][1].im(), -right[0][1].im(), 1e-15);
    assert_eq!(e.left_eigenvectors().unwrap().len(), 2);
    assert!(
        lapack_dgeev(&a, false, false)
            .unwrap()
            .vectors_right
            .is_none()
    );

    // Left eigenvectors of a real spectrum: u' A = lambda u'.
    let m = dense(&[&[4.0, 1.0, 0.0], &[2.0, 3.0, 1.0], &[0.0, 1.0, 2.0]]);
    let e = lapack_dgeev(&m, true, true).unwrap();
    assert!(e.values_imag.iter().all(|&x| x == 0.0));
    let ul = e.vectors_left.as_ref().unwrap();
    for k in 0..3 {
        let u = ul.column(k);
        let uta = blas_dgemv(true, 1.0, &m, u, 0.0, &[0.0; 3]).unwrap();
        assert_vec_close(
            &uta,
            &u.iter().map(|x| e.values_real[k] * x).collect::<Vec<_>>(),
            1e-10,
        );
    }

    // The expert driver agrees, whatever the balancing.
    for balance in [
        DgeevxBalance::None,
        DgeevxBalance::Perm,
        DgeevxBalance::Scale,
        DgeevxBalance::Both,
    ] {
        let x = lapack_dgeevx(balance, &m).unwrap();
        assert_vec_close(
            &sorted(x.eigen.values_real.clone()),
            &sorted(e.values_real.clone()),
            1e-10,
        );
        assert_eq!(x.rconde.len(), 3);
        assert_eq!(x.scale.len(), 3);
        assert!(x.abnrm > 0.0);
        assert!(x.ilo >= 1 && x.ihi <= 3 && x.ilo <= x.ihi);
    }

    // Hessenberg reduction preserves the spectrum.
    let big = random_matrix(6, 6, 21);
    let h = lapack_dgehrd(&big, 1, 6).unwrap();
    for j in 0..6 {
        for i in j + 2..6 {
            assert_eq!(h[(i, j)], 0.0);
        }
    }
    let ev = |m: &Matrix| {
        let e = lapack_dgeev(m, false, false).unwrap();
        let mut v: Vec<(f64, f64)> = e
            .values_real
            .iter()
            .copied()
            .zip(e.values_imag.iter().copied())
            .collect();
        v.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)));
        v
    };
    for (p, q) in ev(&big).iter().zip(ev(&h)) {
        assert_close(p.0, q.0, 1e-9);
        assert_close(p.1, q.1, 1e-9);
    }
    assert!(lapack_dgehrd(&big, 0, 6).is_err());
    assert!(lapack_dgehrd(&big, 3, 2).is_err());
    assert!(lapack_dgehrd(&big, 1, 7).is_err());
    assert_eq!(lapack_dgehrd(&dense(&[&[5.0]]), 1, 1).unwrap()[(0, 0)], 5.0);
    assert!(lapack_dgeev(&Matrix::new(), false, false).is_err());
    assert!(lapack_dgeevx(DgeevxBalance::Both, &Matrix::zeros(2, 1)).is_err());
}

#[test]
fn lapack_dgehrd_preserves_the_spectrum_of_a_random_walk() {
    // tests/unit/igraph_lapack_dgehrd.c: the transposed (column-)stochastic
    // matrix of an undirected ternary tree and its Hessenberg form have the
    // same eigenvalues, in the same order.
    let tree = Graph::kary_tree(10, 3, TreeMode::Undirected).unwrap();
    let sto = tree.get_stochastic(false, None).unwrap().transposed();
    assert_eq!(sto, tree.get_stochastic(true, None).unwrap());
    let hess = lapack_dgehrd(&sto, 1, 10).unwrap();
    let e1 = eigen_matrix(&sto, &EigenWhich::All, EigenAlgorithm::Lapack).unwrap();
    let e2 = eigen_matrix(&hess, &EigenWhich::All, EigenAlgorithm::Lapack).unwrap();
    for (p, q) in e1.values.iter().zip(&e2.values) {
        assert!((p.re() - q.re()).hypot(p.im() - q.im()) < 1e-12);
    }
    // A random walk on a connected bipartite graph: eigenvalues 1 and -1,
    // all real (the walk matrix is similar to a symmetric one).
    assert!(e1.values.iter().all(|z| z.im().abs() < 1e-12));
    let mut re: Vec<f64> = e1.values.iter().map(|z| z.re()).collect();
    re.sort_by(f64::total_cmp);
    assert_close(re[0], -1.0, 1e-12);
    assert_close(re[9], 1.0, 1e-12);
}

// ---------------------------------------------------------------------------
// BLAS
// ---------------------------------------------------------------------------

#[test]
fn blas_dgemm_matches_igraph_unit_test() {
    // tests/unit/igraph_blas_dgemm.c and .out.
    let a = dense(&[&[1.0, 2.0], &[3.0, 4.0]]);
    let b = dense(&[&[5.0, 6.0], &[7.0, 8.0]]);
    let d = dense(&[&[5.0, 6.0, 7.0], &[8.0, 9.0, 10.0]]);
    let e = dense(&[&[5.0, 6.0], &[7.0, 8.0], &[9.0, 10.0]]);
    let rows = |m: Matrix| m.to_rows();
    let gemm =
        |ta, tb, alpha, x: &Matrix, y: &Matrix| blas_dgemm(ta, tb, alpha, x, y, 0.0, None).unwrap();
    assert_eq!(
        rows(gemm(false, false, 1.0, &a, &b)),
        [[19.0, 22.0], [43.0, 50.0]]
    );
    assert_eq!(
        rows(gemm(true, false, 1.0, &a, &b)),
        [[26.0, 30.0], [38.0, 44.0]]
    );
    assert_eq!(
        rows(gemm(false, true, 1.0, &a, &b)),
        [[17.0, 23.0], [39.0, 53.0]]
    );
    assert_eq!(
        rows(gemm(true, true, 1.0, &a, &b)),
        [[23.0, 31.0], [34.0, 46.0]]
    );
    let half = gemm(false, false, 0.5, &a, &b);
    assert_eq!(rows(half.clone()), [[9.5, 11.0], [21.5, 25.0]]);
    // beta = 1 reuses the previous result: 1.5 A B.
    let again = blas_dgemm(false, false, 1.0, &a, &b, 1.0, Some(&half)).unwrap();
    assert_eq!(rows(again), [[28.5, 33.0], [64.5, 75.0]]);
    assert_eq!(
        rows(gemm(false, false, 1.0, &a, &d)),
        [[21.0, 24.0, 27.0], [47.0, 54.0, 61.0]]
    );
    assert_eq!(
        rows(gemm(true, false, 1.0, &d, &a)),
        [[29.0, 42.0], [33.0, 48.0], [37.0, 54.0]]
    );
    // Size mismatches are errors.
    assert!(blas_dgemm(false, false, 1.0, &d, &a, 0.0, None).is_err());
    assert!(blas_dgemm(false, true, 1.0, &a, &d, 0.0, None).is_err());
    assert!(blas_dgemm(false, false, 1.0, &a, &b, 1.0, Some(&d)).is_err());
    assert!(blas_dgemm(false, false, 1.0, &a, &b, 1.0, Some(&e)).is_err());
}

#[test]
fn blas_products_match_naive_ones() {
    let a = random_matrix(4, 3, 31);
    let b = random_matrix(3, 5, 32);
    let c = random_matrix(4, 5, 33);
    let ab = matmul(&a, &b);
    assert_matrix_close(
        &blas_dgemm(false, false, 1.0, &a, &b, 0.0, None).unwrap(),
        &ab,
        1e-12,
    );

    // alpha A B + beta C with a non-square result: igraph checks C's shape
    // against the wrong dimension, the wrapper copes.
    let r = blas_dgemm(false, false, 2.0, &a, &b, -1.0, Some(&c)).unwrap();
    let mut expected = ab.clone();
    for (e, &cv) in expected.as_mut_slice().iter_mut().zip(c.as_slice()) {
        *e = 2.0 * *e - cv;
    }
    assert_matrix_close(&r, &expected, 1e-12);
    // Square case goes through igraph's own beta handling.
    let sq = random_matrix(3, 3, 34);
    let cs = random_matrix(3, 3, 35);
    let r = blas_dgemm(true, true, 1.0, &sq, &sq, 0.5, Some(&cs)).unwrap();
    let mut expected = matmul(&sq.transposed(), &sq.transposed());
    for (e, &cv) in expected.as_mut_slice().iter_mut().zip(cs.as_slice()) {
        *e += 0.5 * cv;
    }
    assert_matrix_close(&r, &expected, 1e-12);
    // Transposition flags.
    assert_matrix_close(
        &blas_dgemm(true, false, 1.0, &a, &a, 0.0, None).unwrap(),
        &matmul(&a.transposed(), &a),
        1e-12,
    );
    assert_matrix_close(
        &blas_dgemm(false, true, 1.0, &b, &b, 0.0, None).unwrap(),
        &matmul(&b, &b.transposed()),
        1e-12,
    );
    // Shape errors.
    assert!(blas_dgemm(false, false, 1.0, &a, &a, 0.0, None).is_err());
    assert!(blas_dgemm(false, false, 1.0, &a, &b, 1.0, Some(&a)).is_err());
    // Degenerate shapes are fine: a 4x0 times 0x5 product is zero.
    let z = blas_dgemm(
        false,
        false,
        1.0,
        &Matrix::zeros(4, 0),
        &Matrix::zeros(0, 5),
        1.0,
        Some(&c),
    )
    .unwrap();
    assert_matrix_close(&z, &c, 0.0);

    // dgemv.
    let x = [1.0, 2.0, -1.0];
    let y = [0.5, 0.5, 0.5, 0.5];
    let naive: Vec<f64> = (0..4)
        .map(|i| 3.0 * (0..3).map(|k| a[(i, k)] * x[k]).sum::<f64>() + 2.0 * y[i])
        .collect();
    assert_vec_close(
        &blas_dgemv(false, 3.0, &a, &x, 2.0, &y).unwrap(),
        &naive,
        1e-12,
    );
    let mut yy = y;
    blas_dgemv_array(false, 3.0, &a, &x, 2.0, &mut yy).unwrap();
    assert_vec_close(&yy, &naive, 1e-12);
    let t = blas_dgemv(true, 1.0, &a, &y, 0.0, &[0.0; 3]).unwrap();
    let naive_t: Vec<f64> = (0..3)
        .map(|j| (0..4).map(|i| a[(i, j)] * y[i]).sum())
        .collect();
    assert_vec_close(&t, &naive_t, 1e-12);
    assert!(blas_dgemv(false, 1.0, &a, &y, 0.0, &y).is_err());
    assert!(blas_dgemv(true, 1.0, &a, &x, 0.0, &x).is_err());
    assert_eq!(
        blas_dgemv(false, 1.0, &Matrix::zeros(2, 0), &[], 3.0, &[1.0, 2.0]).unwrap(),
        vec![3.0, 6.0]
    );

    // Level 1.
    assert_eq!(
        blas_ddot(&[1.0, 2.0, 3.0], &[4.0, -5.0, 6.0]).unwrap(),
        12.0
    );
    assert!(blas_ddot(&[1.0], &[1.0, 2.0]).is_err());
    assert_eq!(blas_ddot(&[], &[]).unwrap(), 0.0);
    assert_close(blas_dnrm2(&[1.0, 2.0, 2.0]).unwrap(), 3.0, 1e-15);
    // dnrm2 avoids overflow.
    assert_close(blas_dnrm2(&[3e200, 4e200]).unwrap() / 1e200, 5.0, 1e-12);
    assert_eq!(blas_dnrm2(&[]).unwrap(), 0.0);
}

// ---------------------------------------------------------------------------
// Spectral embeddings
// ---------------------------------------------------------------------------

/// The directed 4-ary out-tree on 14 vertices used by igraph's unit test.
fn kary_tree_14_4() -> Graph {
    Graph::kary_tree(14, 4, TreeMode::Out).unwrap()
}

#[test]
fn adjacency_embedding_of_a_directed_tree_matches_igraph() {
    // tests/unit/igraph_adjacency_spectral_embedding.c: cvec = degree / 2,
    // four dimensions, LA, unscaled. The first singular vectors are unique
    // up to sign: compare them with igraph's expected output.
    let g = kary_tree_14_4();
    let deg: Vec<f64> = g
        .degree(.., NeighborMode::All, Loops::Twice)
        .unwrap()
        .iter()
        .map(|&d| d as f64 / 2.0)
        .collect();
    let e = g
        .adjacency_spectral_embedding(
            4,
            None,
            EmbeddingWhich::LargestAlgebraic,
            false,
            Some(&deg),
            &ArpackOptions::default(),
        )
        .unwrap();
    assert_eq!(e.x.shape(), (14, 4));
    assert_eq!(e.y.shape(), (14, 4));
    assert_eq!(e.d.len(), 4);
    assert!(
        e.d.windows(2).all(|w| w[0] >= w[1] - 1e-12),
        "singular values are decreasing: {:?}",
        e.d
    );
    let u0 = [
        0.5897, 0.5678, 0.5678, 0.0541, 0.0233, 0.0224, 0.0224, 0.0224, 0.0224, 0.0224, 0.0224,
        0.0224, 0.0224, 0.0021,
    ];
    let v0 = [
        0.3281, 0.5588, 0.5588, 0.1791, 0.1673, 0.1610, 0.1610, 0.1610, 0.1610, 0.1610, 0.1610,
        0.1610, 0.1610, 0.0153,
    ];
    for i in 0..14 {
        assert_close(e.x[(i, 0)].abs(), u0[i], 1e-4);
        assert_close(e.y[(i, 0)].abs(), v0[i], 1e-4);
    }
}

#[test]
fn adjacency_embedding_recovers_planted_blocks() {
    rng::seed(5).unwrap();
    // Three dense blocks of 8 vertices with a few edges between them.
    let mut edges = vec![];
    for b in 0..3 {
        for i in 0..8 {
            for j in i + 1..8 {
                if (i + j) % 5 != 0 {
                    edges.push((8 * b + i, 8 * b + j));
                }
            }
        }
    }
    edges.extend([(0, 8), (9, 17), (18, 2), (5, 20)]);
    let g = Graph::from_edges(&edges, 24, false).unwrap();

    let e = g
        .adjacency_spectral_embedding(
            6,
            None,
            EmbeddingWhich::LargestAlgebraic,
            true,
            None,
            &ArpackOptions::default(),
        )
        .unwrap();
    // Three large eigenvalues, then a gap: dimensionality selection finds 3.
    assert!(e.d[2] > 3.0 * e.d[3].abs(), "{:?}", e.d);
    assert_eq!(dim_select(&e.d).unwrap(), 3);

    // Vertices of the same block are much closer than vertices of different
    // blocks in the 3-dimensional embedding.
    let e3 = g
        .adjacency_spectral_embedding(
            3,
            None,
            EmbeddingWhich::LargestAlgebraic,
            true,
            None,
            &ArpackOptions::default(),
        )
        .unwrap();
    let dist = |a: usize, b: usize| -> f64 {
        e3.x.row(a)
            .iter()
            .zip(e3.x.row(b))
            .map(|(p, q)| (p - q).powi(2))
            .sum::<f64>()
            .sqrt()
    };
    for block in 0..3 {
        for other in 0..3 {
            let d = dist(8 * block + 3, 8 * other + 4);
            if block == other {
                assert!(d < 0.5, "within block {block}: {d}");
            } else {
                assert!(d > 0.8, "between {block} and {other}: {d}");
            }
        }
    }
    // Undirected: Y equals X.
    assert_eq!(e3.x, e3.y);

    // Weighted version: doubling all the weights scales the eigenvalues.
    let w = vec![2.0; g.ecount()];
    let ew = g
        .adjacency_spectral_embedding(
            3,
            Some(&w),
            EmbeddingWhich::LargestAlgebraic,
            true,
            None,
            &ArpackOptions::default(),
        )
        .unwrap();
    assert_vec_close(
        &ew.d,
        &e3.d.iter().map(|x| 2.0 * x).collect::<Vec<_>>(),
        1e-8,
    );
    // A scalar cvec shifts the spectrum.
    let shifted = g
        .adjacency_spectral_embedding(
            3,
            None,
            EmbeddingWhich::LargestAlgebraic,
            true,
            Some(&[1.0]),
            &ArpackOptions::default(),
        )
        .unwrap();
    assert_vec_close(
        &shifted.d,
        &e3.d.iter().map(|x| x + 1.0).collect::<Vec<_>>(),
        1e-8,
    );
}

#[test]
fn embedding_argument_errors() {
    let g = cycle(6);
    let o = ArpackOptions::default();
    let la = EmbeddingWhich::LargestAlgebraic;
    assert!(
        g.adjacency_spectral_embedding(0, None, la, true, None, &o)
            .is_err()
    );
    assert!(
        g.adjacency_spectral_embedding(7, None, la, true, None, &o)
            .is_err()
    );
    assert!(
        g.adjacency_spectral_embedding(2, Some(&[1.0]), la, true, None, &o)
            .is_err()
    );
    assert!(
        g.adjacency_spectral_embedding(2, None, la, true, Some(&[1.0, 2.0]), &o)
            .is_err()
    );
    assert!(
        g.laplacian_spectral_embedding(2, None, la, LaplacianEmbeddingType::OAP, true, &o)
            .is_err()
    );
    let d = Graph::from_edges(&[(0, 1), (1, 2), (2, 0)], 3, true).unwrap();
    assert!(
        d.laplacian_spectral_embedding(1, None, la, LaplacianEmbeddingType::DA, true, &o)
            .is_err()
    );
    assert!(dim_select(&[]).is_err());
    assert!(dim_select(&[1.0, f64::NAN]).is_err());
    // Without any gap every split is equally (un)likely: one group.
    assert_eq!(dim_select(&[1.0, 1.0]).unwrap(), 2);
    assert_eq!(dim_select(&[0.5; 6]).unwrap(), 6);
    // Two clear groups.
    assert_eq!(dim_select(&[9.0, 8.0, 1.0, 1.0, 1.0]).unwrap(), 2);
    // A graph without edges embeds to the origin.
    let empty = Graph::new(4, false);
    let e = empty
        .adjacency_spectral_embedding(2, None, la, true, None, &o)
        .unwrap();
    assert!(e.x.as_slice().iter().all(|&x| x == 0.0));
}

#[test]
fn laplacian_embedding_fiedler_vector_of_a_path_is_monotone() {
    rng::seed(9).unwrap();
    let n = 12;
    let g = path(n as i64);
    let e = g
        .laplacian_spectral_embedding(
            2,
            None,
            EmbeddingWhich::SmallestAlgebraic,
            LaplacianEmbeddingType::DA,
            false,
            &ArpackOptions::default(),
        )
        .unwrap();
    let spectrum = path_laplacian_spectrum(n);
    assert_vec_close(&sorted(e.d.clone()), &spectrum[..2], 1e-8);
    // The Fiedler vector (eigenvalue 2 - 2cos(pi/n)) is a discrete cosine:
    // monotone along the path.
    let k = if (e.d[0] - spectrum[1]).abs() < 1e-8 {
        0
    } else {
        1
    };
    let f = e.x.column(k);
    let increasing = f.windows(2).all(|w| w[0] < w[1]);
    let decreasing = f.windows(2).all(|w| w[0] > w[1]);
    assert!(increasing || decreasing, "{f:?}");

    // Normalized Laplacians: I - D^-1/2 A D^-1/2 has eigenvalues in [0, 2];
    // D^-1/2 A D^-1/2 has largest eigenvalue 1.
    let idad = g
        .laplacian_spectral_embedding(
            2,
            None,
            EmbeddingWhich::LargestAlgebraic,
            LaplacianEmbeddingType::IDAD,
            true,
            &ArpackOptions::default(),
        )
        .unwrap();
    assert!(idad.d.iter().all(|&x| (-1e-9..=2.0 + 1e-9).contains(&x)));
    // They are the two largest eigenvalues of the symmetric normalized
    // Laplacian built by the structural module.
    let sym = g
        .get_laplacian(
            NeighborMode::All,
            igraph::structural::LaplacianNormalization::Symmetric,
            None,
        )
        .unwrap();
    let dense = lapack_dsyevr(&sym, &SymmetricRange::Select(n - 2..n), 1e-12).unwrap();
    assert_vec_close(&sorted(idad.d.clone()), &dense.values, 1e-8);
    let dad = g
        .laplacian_spectral_embedding(
            1,
            None,
            EmbeddingWhich::LargestAlgebraic,
            LaplacianEmbeddingType::DAD,
            true,
            &ArpackOptions::default(),
        )
        .unwrap();
    assert_close(dad.d[0], 1.0, 1e-8);

    // Directed graphs use O^-1/2 A P^-1/2: a directed cycle has all singular values 1.
    let edges: Vec<(i64, i64)> = (0..8).map(|i| (i, (i + 1) % 8)).collect();
    let dc = Graph::from_edges(&edges, 8, true).unwrap();
    let oap = dc
        .laplacian_spectral_embedding(
            2,
            None,
            EmbeddingWhich::LargestAlgebraic,
            LaplacianEmbeddingType::OAP,
            false,
            &ArpackOptions::default(),
        )
        .unwrap();
    assert_vec_close(&oap.d, &[1.0, 1.0], 1e-8);
    assert_eq!(oap.y.shape(), (8, 2));
}

// ---------------------------------------------------------------------------
// Use cases
// ---------------------------------------------------------------------------

/// Story: Zachary's karate club split into two factions after a dispute
/// between the instructor (vertex 0) and the administrator (vertex 33).
/// Spectral bisection — the sign pattern of the Fiedler vector of the
/// normalized Laplacian — predicts the split from the friendship network
/// alone, and does so identically with three different tools: the
/// Laplacian embedding, ARPACK on a sparse matrix and dense LAPACK.
#[test]
fn use_case_spectral_bisection_of_the_karate_club() {
    rng::seed(2024).unwrap();
    let g = Graph::famous("Zachary").unwrap();
    let n = g.vcount();
    // Same network (and vertex numbering) as the shared test fixture.
    let normalized = |g: &Graph| {
        let mut e: Vec<(i64, i64)> = g
            .edge_list()
            .iter()
            .map(|&(a, b)| (a.min(b), a.max(b)))
            .collect();
        e.sort_unstable();
        e
    };
    assert_eq!(normalized(&g), normalized(&karate()));
    let instructor_faction = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 16, 17, 19, 21];

    // 1. Laplacian spectral embedding: the two smallest eigenvectors of D - A.
    let e = g
        .laplacian_spectral_embedding(
            2,
            None,
            EmbeddingWhich::SmallestAlgebraic,
            LaplacianEmbeddingType::DA,
            false,
            &ArpackOptions::default(),
        )
        .unwrap();
    let k = if e.d[0] > e.d[1] { 0 } else { 1 };
    let fiedler_embedding: Vec<f64> = e.x.column(k).to_vec();

    // 2. The same Laplacian as a sparse matrix (from the structural module),
    //    solved with ARPACK.
    let triplets: Vec<(usize, usize, f64)> = g
        .get_laplacian_sparse(
            NeighborMode::All,
            igraph::structural::LaplacianNormalization::Unnormalized,
            None,
        )
        .unwrap()
        .into_iter()
        .map(|(i, j, x)| (i as usize, j as usize, x))
        .collect();
    let lap = SparseMat::from_triplets(n, n, &triplets)
        .unwrap()
        .compress()
        .unwrap();
    assert!(lap.is_symmetric().unwrap());
    // It agrees with the Laplacian assembled edge by edge.
    let edges: Vec<(usize, usize)> = g
        .edge_list()
        .iter()
        .map(|&(a, b)| (a as usize, b as usize))
        .collect();
    let by_hand = SparseMat::from_triplets(n, n, &laplacian_triplets(n, &edges))
        .unwrap()
        .compress()
        .unwrap();
    assert_eq!(lap.to_dense().unwrap(), by_hand.to_dense().unwrap());
    let degrees = lap.rowsums().unwrap();
    assert!(
        degrees.iter().all(|&s| s.abs() < 1e-12),
        "Laplacian rows sum to zero"
    );
    let arpack = lap
        .arpack_rssolve(
            &ArpackOptions::default()
                .with_nev(2)
                .with_which(ArpackWhich::SmallestAlgebraic)
                .with_ncv(20),
            SparseSolveMethod::Lu,
        )
        .unwrap();
    let idx = if arpack.values[0] > arpack.values[1] {
        0
    } else {
        1
    };
    let fiedler_arpack = arpack.vectors.column(idx).to_vec();

    // 3. Dense LAPACK as the reference.
    let dense_l = lap.to_dense().unwrap();
    let lapack = lapack_dsyevr(&dense_l, &SymmetricRange::Select(0..2), 1e-12).unwrap();
    assert_close(lapack.values[0], 0.0, 1e-10);
    let fiedler_value = lapack.values[1];
    let fiedler_lapack = lapack.vectors.column(1).to_vec();
    assert_close(sorted(arpack.values.clone())[1], fiedler_value, 1e-8);
    assert_close(
        e.d.iter().copied().fold(f64::MIN, f64::max),
        fiedler_value,
        1e-8,
    );

    // The three Fiedler vectors agree up to sign.
    let agree = |a: &[f64], b: &[f64]| {
        let dot = blas_ddot(a, b).unwrap();
        (dot.abs() - blas_dnrm2(a).unwrap() * blas_dnrm2(b).unwrap()).abs() < 1e-6
    };
    assert!(agree(&fiedler_embedding, &fiedler_lapack));
    assert!(agree(&fiedler_arpack, &fiedler_lapack));

    // Split by sign, orient so that the instructor is on the positive side.
    let sign = fiedler_lapack[0].signum();
    let predicted: Vec<bool> = fiedler_lapack.iter().map(|&x| x * sign > 0.0).collect();
    assert!(
        predicted[0] && !predicted[33],
        "the two leaders end up apart"
    );
    let correct = (0..n)
        .filter(|v| predicted[*v] == instructor_faction.contains(v))
        .count();
    assert!(
        correct >= 32,
        "spectral bisection recovers the split: {correct}/34"
    );
}

/// Story: PageRank as the dominant eigenvector of the Google matrix of a tiny
/// web, computed matrix-free with the non-symmetric ARPACK solver and
/// verified as a fixed point with sparse matrix algebra.
#[test]
fn use_case_pagerank_via_arpack() {
    rng::seed(3).unwrap();
    // Links i -> j.
    let links = [
        (0, 1),
        (0, 2),
        (1, 2),
        (2, 0),
        (3, 2),
        (4, 3),
        (4, 5),
        (5, 4),
        (5, 0),
    ];
    let n = 6;
    let damping = 0.85;
    // Column-stochastic transition matrix M[j, i] = 1 / outdeg(i).
    let mut m = SparseMat::new(n, n).unwrap();
    for &(i, j) in &links {
        m.entry(j, i, 1.0).unwrap();
    }
    m.normalize_cols(false).unwrap();
    assert_vec_close(&m.colsums().unwrap(), &[1.0; 6], 1e-12);
    let m = m.compress().unwrap();

    // Google matrix G = d M + (1 - d)/n 11', applied without forming it.
    let google = |x: &[f64], y: &mut [f64]| {
        let mx = m.mul_vec(x).unwrap();
        let s: f64 = x.iter().sum();
        for i in 0..n {
            y[i] = damping * mx[i] + (1.0 - damping) / n as f64 * s;
        }
    };
    let res = arpack_rnsolve(n, google, &ArpackOptions::default().with_nev(1), None).unwrap();
    // Perron-Frobenius: the dominant eigenvalue of a stochastic matrix is 1.
    assert_close(res.values[0].re(), 1.0, 1e-10);
    assert_close(res.values[0].im(), 0.0, 1e-10);
    let raw: Vec<f64> = res.vectors[0].iter().map(|z| z.re()).collect();
    let total: f64 = raw.iter().sum();
    let pagerank: Vec<f64> = raw.iter().map(|x| x / total).collect();
    assert!(pagerank.iter().all(|&p| p > 0.0));
    assert_close(pagerank.iter().sum(), 1.0, 1e-12);

    // Fixed point: d M p + (1 - d)/n = p.
    let mp = m.mul_vec(&pagerank).unwrap();
    for i in 0..n {
        assert_close(
            damping * mp[i] + (1.0 - damping) / n as f64,
            pagerank[i],
            1e-10,
        );
    }
    // The centrality module (PRPACK) agrees.
    let web_links: Vec<(i64, i64)> = links.iter().map(|&(i, j)| (i as i64, j as i64)).collect();
    let web = Graph::from_edges(&web_links, n, true).unwrap();
    let prpack = web
        .pagerank(None, .., &igraph::centrality::PageRankOptions::default())
        .unwrap();
    assert_vec_close(&prpack.scores, &pagerank, 1e-10);

    // Page 2 collects links from 0, 1 and 3: it ranks first.
    let best = (0..n)
        .max_by(|&a, &b| pagerank[a].total_cmp(&pagerank[b]))
        .unwrap();
    assert_eq!(best, 2);
}

/// igraph solves 2 x 2 problems with a closed-form shortcut instead of
/// ARPACK; in igraph 1.0.0 and 1.0.1 it treats "largest (smallest)
/// magnitude" as "largest (smallest) algebraic" and returns unnormalized
/// eigenvectors. The wrappers must give ARPACK's answers.
#[test]
fn two_by_two_problems_follow_the_arpack_conventions() {
    // diag(1, -3): the eigenvalue of largest magnitude is -3.
    let diag = |x: &[f64], y: &mut [f64]| {
        y[0] = x[0];
        y[1] = -3.0 * x[1];
    };
    let run = |which: ArpackWhich, nev: usize| {
        let opts = ArpackOptions::default().with_nev(nev).with_which(which);
        arpack_rssolve(2, diag, &opts, None).unwrap()
    };
    assert_eq!(run(ArpackWhich::LargestMagnitude, 1).values, vec![-3.0]);
    assert_eq!(run(ArpackWhich::SmallestMagnitude, 1).values, vec![1.0]);
    assert_eq!(run(ArpackWhich::LargestAlgebraic, 1).values, vec![1.0]);
    assert_eq!(run(ArpackWhich::SmallestAlgebraic, 1).values, vec![-3.0]);
    assert_eq!(
        run(ArpackWhich::LargestMagnitude, 2).values,
        vec![-3.0, 1.0]
    );
    assert_eq!(
        run(ArpackWhich::SmallestMagnitude, 2).values,
        vec![1.0, -3.0]
    );
    assert_eq!(run(ArpackWhich::BothEnds, 1).values, vec![1.0]);
    assert_eq!(run(ArpackWhich::BothEnds, 2).values, vec![-3.0, 1.0]);
    let lm = run(ArpackWhich::LargestMagnitude, 1);
    assert_eq!(lm.vectors.shape(), (2, 1));
    assert_eq!(lm.info.nconv, 1);
    assert_close(lm.vectors[(0, 0)], 0.0, 1e-15);
    assert_close(lm.vectors[(1, 0)].abs(), 1.0, 1e-15);

    // [[2, 1], [1, 2]]: eigenvalues 3 and 1 with unit eigenvectors
    // (1, 1)/sqrt(2) and (1, -1)/sqrt(2).
    let two = |x: &[f64], y: &mut [f64]| {
        y[0] = 2.0 * x[0] + x[1];
        y[1] = x[0] + 2.0 * x[1];
    };
    let opts = ArpackOptions::default()
        .with_nev(2)
        .with_which(ArpackWhich::LargestAlgebraic);
    let res = arpack_rssolve(2, two, &opts, None).unwrap();
    let h = std::f64::consts::FRAC_1_SQRT_2;
    for j in 0..2 {
        let v = res.vectors.column(j);
        assert_close(v[0].abs(), h, 1e-14);
        assert_close(v[1].abs(), h, 1e-14);
    }
    assert!(res.vectors[(0, 0)] * res.vectors[(1, 0)] > 0.0);
    assert!(res.vectors[(0, 1)] * res.vectors[(1, 1)] < 0.0);

    // The same through a sparse matrix and through the eigen front-end.
    let sparse = SparseMat::from_triplets(2, 2, &[(0, 0, 1.0), (1, 1, -3.0)]).unwrap();
    let opts = ArpackOptions::default().with_which(ArpackWhich::LargestMagnitude);
    let res = sparse.arpack_rssolve(&opts, SparseSolveMethod::Lu).unwrap();
    assert_eq!(res.values, vec![-3.0]);
    assert_close(res.vectors[(1, 0)].abs(), 1.0, 1e-15);
    let dense = sparse.to_dense().unwrap();
    for algorithm in [
        EigenAlgorithm::Arpack,
        EigenAlgorithm::Auto,
        EigenAlgorithm::Lapack,
    ] {
        let e = eigen_matrix_symmetric(
            &dense,
            &EigenWhich::LargestMagnitude(1),
            algorithm,
            &ArpackOptions::default(),
        )
        .unwrap();
        assert_eq!(e.values, vec![-3.0], "{algorithm:?}");
    }

    // A graph with two vertices: K2 plus a self-loop on vertex 0 has
    // adjacency [[1, 1], [1, 0]], eigenvalues (1 +- sqrt(5)) / 2.
    let g = Graph::from_edges(&[(0, 1), (0, 0)], 2, false).unwrap();
    let e = g
        .eigen_adjacency(
            &EigenWhich::LargestAlgebraic(1),
            EigenAlgorithm::Auto,
            &ArpackOptions::default(),
        )
        .unwrap();
    let phi = (1.0 + 5f64.sqrt()) / 2.0;
    assert_close(e.values[0], phi, 1e-12);
    let v = e.vectors.column(0);
    assert_close(v[0] * v[0] + v[1] * v[1], 1.0, 1e-12);
    assert_close(v[0] / v[1], phi, 1e-12);

    // Non-symmetric 2 x 2 problems get unit eigenvectors too.
    let res = arpack_rnsolve(2, two, &ArpackOptions::default().with_nev(2), None).unwrap();
    let rot = |x: &[f64], y: &mut [f64]| {
        y[0] = -x[1];
        y[1] = x[0];
    };
    let rres = arpack_rnsolve(2, rot, &ArpackOptions::default().with_nev(2), None).unwrap();
    for v in res.vectors.iter().chain(&rres.vectors) {
        let norm2: f64 = v.iter().map(|c| c.re() * c.re() + c.im() * c.im()).sum();
        assert_close(norm2, 1.0, 1e-14);
    }
    assert_close(rres.values[0].im().abs(), 1.0, 1e-14);
}

/// With `Auto`, igraph itself would pick ARPACK for `All`, `Interval` and
/// `Select` on matrices of order 100 or more, where they are not supported.
#[test]
fn auto_algorithm_uses_lapack_where_arpack_cannot_work() {
    let n = 120;
    let a = path_laplacian(n).to_dense().unwrap();
    let opts = ArpackOptions::default();
    let all = eigen_matrix_symmetric(&a, &EigenWhich::All, EigenAlgorithm::Auto, &opts).unwrap();
    assert_vec_close(&all.values, &path_laplacian_spectrum(n), 1e-10);
    let sel =
        eigen_matrix_symmetric(&a, &EigenWhich::Select(0..3), EigenAlgorithm::Auto, &opts).unwrap();
    assert_vec_close(&sel.values, &path_laplacian_spectrum(n)[..3], 1e-10);
    let iv = eigen_matrix_symmetric(
        &a,
        &EigenWhich::Interval {
            low: 3.99,
            high: 4.0,
        },
        EigenAlgorithm::Auto,
        &opts,
    )
    .unwrap();
    let expected: Vec<f64> = path_laplacian_spectrum(n)
        .into_iter()
        .filter(|&x| x > 3.99)
        .collect();
    assert!(!expected.is_empty());
    assert_vec_close(&iv.values, &expected, 1e-10);
    // Explicitly asking ARPACK for them is still igraph's error.
    let err = eigen_matrix_symmetric(&a, &EigenWhich::Select(0..3), EigenAlgorithm::Arpack, &opts)
        .unwrap_err();
    assert_eq!(err.kind(), ErrorKind::Unimplemented);
    // A few extreme eigenvalues of a large matrix still go to ARPACK.
    rng::seed(7).unwrap();
    let top = eigen_matrix_symmetric(
        &a,
        &EigenWhich::LargestAlgebraic(2),
        EigenAlgorithm::Auto,
        &opts,
    )
    .unwrap();
    assert_vec_close(
        &top.values,
        &[
            path_laplacian_spectrum(n)[n - 1],
            path_laplacian_spectrum(n)[n - 2],
        ],
        1e-8,
    );
}

/// The tie-breaking rules of `eigen_matrix`: `i`, `-i` and `1` all have
/// magnitude 1.
#[test]
fn eigen_matrix_orders_ties_between_real_and_complex_eigenvalues() {
    let a = dense(&[&[0.0, -1.0, 0.0], &[1.0, 0.0, 0.0], &[0.0, 0.0, 1.0]]);
    let pairs = |which: EigenWhich| -> Vec<(f64, f64)> {
        eigen_matrix(&a, &which, EigenAlgorithm::Lapack)
            .unwrap()
            .values
            .iter()
            .map(|c| (c.re().round(), c.im().round()))
            .collect()
    };
    // Largest magnitude: real first, then larger imaginary part.
    assert_eq!(
        pairs(EigenWhich::LargestMagnitude(3)),
        vec![(1.0, 0.0), (0.0, 1.0), (0.0, -1.0)]
    );
    // Smallest magnitude and All: exactly the reverse.
    let reverse = vec![(0.0, -1.0), (0.0, 1.0), (1.0, 0.0)];
    assert_eq!(pairs(EigenWhich::SmallestMagnitude(3)), reverse);
    assert_eq!(pairs(EigenWhich::All), reverse);
    assert_eq!(pairs(EigenWhich::Select(2..3)), vec![(1.0, 0.0)]);
    // `Auto` and ARPACK are not implemented for general matrices.
    for algorithm in [EigenAlgorithm::Auto, EigenAlgorithm::Arpack] {
        let err = eigen_matrix(&a, &EigenWhich::All, algorithm).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::Unimplemented);
    }
}

// ---------------------------------------------------------------------------
// Audit regressions: failing igraph calls inside callbacks, nested ARPACK
// runs, singular operators.
// ---------------------------------------------------------------------------

/// Makes a failing igraph call (an edge to a missing vertex): igraph's error
/// handler then frees the current level of its "finally" stack.
fn failing_igraph_call() {
    let mut g = Graph::empty(2, false).unwrap();
    assert_eq!(
        g.add_edge(0, 99).unwrap_err().kind(),
        ErrorKind::InvalidVertexId
    );
}

fn finally_stack_size() -> i32 {
    unsafe { igraph::ffi::IGRAPH_FINALLY_STACK_SIZE() }
}

/// `y = A x` for the adjacency matrix of the cycle `C_n`.
fn cycle_matvec(n: usize) -> impl Fn(&[f64], &mut [f64]) {
    move |x: &[f64], y: &mut [f64]| {
        for i in 0..n {
            y[i] = x[(i + 1) % n] + x[(i + n - 1) % n];
        }
    }
}

fn top2_opts() -> ArpackOptions {
    ArpackOptions::default()
        .with_nev(2)
        .with_which(ArpackWhich::LargestAlgebraic)
}

#[test]
fn failing_igraph_call_in_arpack_matvec_leaves_the_solver_intact() {
    let n = 12;
    igraph::rng::seed(5).unwrap();
    let expected = arpack_rssolve(n, cycle_matvec(n), &top2_opts(), None).unwrap();
    let mut calls = 0;
    igraph::rng::seed(5).unwrap();
    let res = arpack_rssolve(
        n,
        |x: &[f64], y: &mut [f64]| {
            calls += 1;
            if calls <= 3 {
                failing_igraph_call();
            }
            cycle_matvec(n)(x, y)
        },
        &top2_opts(),
        None,
    )
    .unwrap();
    assert_vec_close(&res.values, &expected.values, 1e-12);
    assert_vec_close(&res.values, &[2.0, 2.0 * (PI / 6.0).cos()], 1e-8);
    assert_eq!(finally_stack_size(), 0);
    // Same through the non-symmetric solver and the eigen front-end.
    let res = arpack_rnsolve(
        n,
        |x: &[f64], y: &mut [f64]| {
            failing_igraph_call();
            cycle_matvec(n)(x, y)
        },
        &ArpackOptions::default().with_which(ArpackWhich::LargestReal),
        None,
    )
    .unwrap();
    assert!((res.values[0].re() - 2.0).abs() < 1e-8);
    let e = eigen_symmetric_fn(
        n,
        |x: &[f64], y: &mut [f64]| {
            failing_igraph_call();
            cycle_matvec(n)(x, y)
        },
        &EigenWhich::LargestAlgebraic(1),
        EigenAlgorithm::Arpack,
        &ArpackOptions::default(),
    )
    .unwrap();
    assert!((e.values[0] - 2.0).abs() < 1e-8);
    assert_eq!(finally_stack_size(), 0);
}

#[test]
fn failing_igraph_call_in_fkeep_leaves_the_matrix_intact() {
    let mut m = SparseMat::from_dense(
        &dense(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], &[7.0, 8.0, 9.0]]),
        0.0,
    )
    .unwrap();
    m.fkeep(|i, j, _| {
        failing_igraph_call();
        i <= j
    })
    .unwrap();
    assert_eq!(
        sorted(m.triplets().iter().map(|t| t.2).collect()),
        vec![1.0, 2.0, 3.0, 5.0, 6.0, 9.0]
    );
    assert_eq!(finally_stack_size(), 0);
}

#[test]
fn nested_arpack_runs_in_matvec_are_refused() {
    let n = 12;
    igraph::rng::seed(9).unwrap();
    let expected = arpack_rssolve(n, cycle_matvec(n), &top2_opts(), None).unwrap();
    let mut nested = Vec::new();
    let mut calls = 0;
    igraph::rng::seed(9).unwrap();
    let res = arpack_rssolve(
        n,
        |x: &[f64], y: &mut [f64]| {
            calls += 1;
            if calls <= 3 {
                // Every ARPACK-based entry point of the module is refused.
                let m = 200;
                nested.push(arpack_rssolve(m, cycle_matvec(m), &top2_opts(), None).err());
                nested.push(
                    arpack_rnsolve(m, cycle_matvec(m), &ArpackOptions::default(), None).err(),
                );
                nested.push(
                    eigen_symmetric_fn(
                        m,
                        cycle_matvec(m),
                        &EigenWhich::LargestAlgebraic(1),
                        EigenAlgorithm::Arpack,
                        &ArpackOptions::default(),
                    )
                    .err(),
                );
                let ring = Graph::ring(m, false, false, true).unwrap();
                nested.push(
                    ring.eigen_adjacency(
                        &EigenWhich::LargestAlgebraic(1),
                        EigenAlgorithm::Arpack,
                        &ArpackOptions::default(),
                    )
                    .err(),
                );
                nested.push(
                    ring.adjacency_spectral_embedding(
                        2,
                        None,
                        EmbeddingWhich::LargestAlgebraic,
                        true,
                        None,
                        &ArpackOptions::default(),
                    )
                    .err(),
                );
                let lap = path_laplacian(m);
                nested.push(
                    lap.arpack_rssolve(&top2_opts(), SparseSolveMethod::Lu)
                        .err(),
                );
                // LAPACK does not touch ARPACK's state: allowed.
                let lapack = eigen_symmetric_fn(
                    4,
                    cycle_matvec(4),
                    &EigenWhich::All,
                    EigenAlgorithm::Lapack,
                    &ArpackOptions::default(),
                );
                assert!(lapack.is_ok());
            }
            cycle_matvec(n)(x, y)
        },
        &top2_opts(),
        None,
    )
    .unwrap();
    // The outer run is unaffected by the refused nested runs.
    assert_vec_close(&res.values, &expected.values, 1e-12);
    assert_eq!(nested.len(), 18);
    for err in nested {
        let err = err.expect("nested ARPACK run must fail");
        assert_eq!(err.kind(), ErrorKind::Failure);
        assert!(err.message().contains("cannot be nested"), "{err}");
    }
    // The guard is released: ARPACK works again at top level.
    let again = arpack_rssolve(200, cycle_matvec(200), &top2_opts(), None).unwrap();
    assert!((again.values[0] - 2.0).abs() < 1e-8);
}

#[test]
fn nested_arpack_runs_in_interruption_handler_are_refused() {
    let n = 60;
    igraph::rng::seed(2).unwrap();
    let expected = arpack_rssolve(n, cycle_matvec(n), &top2_opts(), None).unwrap();
    let outcomes = Rc::new(RefCell::new(Vec::new()));
    let sink = Rc::clone(&outcomes);
    let res = igraph::misc::with_interruption_handler(
        move || {
            let m = 30;
            let r = arpack_rssolve(m, cycle_matvec(m), &top2_opts(), None);
            sink.borrow_mut().push(r.map(|_| ()).map_err(|e| e.kind()));
            false
        },
        || {
            igraph::rng::seed(2).unwrap();
            arpack_rssolve(n, cycle_matvec(n), &top2_opts(), None)
        },
    )
    .unwrap();
    assert_vec_close(&res.values, &expected.values, 1e-12);
    let outcomes = outcomes.borrow();
    assert!(
        !outcomes.is_empty(),
        "ARPACK polls the interruption handler"
    );
    assert!(outcomes.iter().all(|o| *o == Err(ErrorKind::Failure)));
    // Outside of an ARPACK run the same handler may use ARPACK.
    let ok = Rc::new(Cell::new(false));
    let flag = Rc::clone(&ok);
    let _guard = igraph::misc::set_interruption_handler(move || {
        flag.set(arpack_rssolve(30, cycle_matvec(30), &top2_opts(), None).is_ok());
        false
    });
    assert!(!igraph::misc::allow_interruption());
    assert!(ok.get());
}

#[test]
fn arpack_guard_is_released_after_a_panicking_matvec() {
    let r = std::panic::catch_unwind(|| {
        arpack_rssolve(
            12,
            |_: &[f64], _: &mut [f64]| panic!("boom"),
            &top2_opts(),
            None,
        )
    });
    assert!(r.is_err());
    let res = arpack_rssolve(12, cycle_matvec(12), &top2_opts(), None).unwrap();
    assert!((res.values[0] - 2.0).abs() < 1e-8);
}

#[test]
fn arpack_misses_the_null_space_of_a_singular_operator_unless_shifted() {
    // Path Laplacian on 10 vertices plus `shift` times the identity.
    let n = 10;
    let laplacian = move |shift: f64| {
        move |x: &[f64], y: &mut [f64]| {
            for i in 0..n {
                let degree = if i == 0 || i == n - 1 { 1.0 } else { 2.0 };
                y[i] = (degree + shift) * x[i];
                if i > 0 {
                    y[i] -= x[i - 1];
                }
                if i + 1 < n {
                    y[i] -= x[i + 1];
                }
            }
        }
    };
    let spectrum: Vec<f64> = (0..n)
        .map(|k| 2.0 - 2.0 * (PI * k as f64 / n as f64).cos())
        .collect();
    // The ramp (0, 1, ..., n - 1) has a constant component, but
    // L ramp = e_{n-1} - e_0 has none (and is antisymmetric under the
    // reflection of the path, which L preserves): ARPACK, which iterates
    // from L v0, never sees the null vector and returns the eigenvalues of
    // the antisymmetric eigenvectors, k = 1 and k = 3.
    let ramp: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let opts = ArpackOptions::default()
        .with_nev(2)
        .with_which(ArpackWhich::SmallestMagnitude)
        .with_start(ramp);
    let plain = arpack_rssolve(n, laplacian(0.0), &opts, None).unwrap();
    assert_vec_close(
        &sorted(plain.values.clone()),
        &[spectrum[1], spectrum[3]],
        1e-8,
    );
    // With the shift, the null vector is found (and is constant).
    for shift in [0.5, 1.0, 3.0] {
        let shifted = arpack_rssolve(
            n,
            laplacian(shift),
            &opts.clone().with_which(ArpackWhich::SmallestAlgebraic),
            None,
        )
        .unwrap();
        let values: Vec<f64> = shifted.values.iter().map(|v| v - shift).collect();
        assert_vec_close(&sorted(values), &spectrum[..2], 1e-8);
        let v0 = shifted.vectors.column(0);
        assert!(
            v0.iter()
                .all(|x| (x.abs() - 1.0 / (n as f64).sqrt()).abs() < 1e-8)
        );
    }
    // A start vector inside the null space fails outright.
    let err = arpack_rssolve(
        n,
        laplacian(0.0),
        &opts.clone().with_start(vec![1.0; n]),
        None,
    )
    .unwrap_err();
    assert_eq!(ArpackError::from_error(&err), Some(ArpackError::ZeroStart));
}

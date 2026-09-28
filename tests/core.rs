//! Integration tests for the core of the bindings: errors, containers,
//! selectors, the graph interface and random numbers.

mod common;

use common::*;
use igraph::bitset::Bitset;
use igraph::error::{self, catch_panic, check, strerror, take_warnings};
use igraph::igraph_complex_t as Complex;
use igraph::prelude::*;
use igraph::vector::{format_real, format_real_precise};
use igraph::{ffi, igraph_call};
use std::cmp::Ordering;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[test]
fn error_kinds_round_trip_through_raw_codes() {
    let kinds = [
        ErrorKind::Failure,
        ErrorKind::OutOfMemory,
        ErrorKind::Parse,
        ErrorKind::InvalidValue,
        ErrorKind::Exists,
        ErrorKind::InvalidVertexId,
        ErrorKind::InvalidEdgeId,
        ErrorKind::InvalidMode,
        ErrorKind::File,
        ErrorKind::Unimplemented,
        ErrorKind::Interrupted,
        ErrorKind::Diverged,
        ErrorKind::Arpack,
        ErrorKind::NegativeCycle,
        ErrorKind::Internal,
        ErrorKind::AttributeCombination,
        ErrorKind::Overflow,
        ErrorKind::Underflow,
        ErrorKind::RandomWalkStuck,
        ErrorKind::Stop,
        ErrorKind::Range,
        ErrorKind::NoSolution,
    ];
    for kind in kinds {
        assert_eq!(ErrorKind::from_raw(kind.to_raw()), kind);
        assert!(!kind.description().is_empty(), "{kind:?}");
        // Errors built on the Rust side carry the matching code.
        let e = Error::new(kind, "boom");
        assert_eq!(e.code(), kind.to_raw());
        assert_eq!(e.kind(), kind);
        assert!(e.to_string().ends_with(": boom"));
    }
    assert_eq!(ErrorKind::from_raw(9999), ErrorKind::Other(9999));
    assert_eq!(
        strerror(ffi::igraph_error_type_t_IGRAPH_SUCCESS),
        "No error"
    );
    assert_eq!(
        ErrorKind::InvalidVertexId.to_string(),
        ErrorKind::InvalidVertexId.description()
    );
}

#[test]
fn igraph_errors_carry_kind_message_and_location() {
    let g = path(3);
    let err = g.edge(10).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidEdgeId);
    assert!(!err.message().is_empty());
    assert!(err.file().ends_with(".c"), "{}", err.file());
    assert!(err.line() > 0);
    let text = err.to_string();
    assert!(
        text.contains(err.message()) && text.contains(err.file()),
        "{text}"
    );

    let err = g
        .degree(&[0, 5], NeighborMode::All, Loops::Twice)
        .unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidVertexId);
    let err = Graph::from_flat_edges(&[0, 1, 2], 3, false).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidValue);
    let err = Graph::from_edges(&[(0, -1)], 3, false).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidVertexId);
}

#[test]
fn a_failure_does_not_leak_its_message_into_the_next_error() {
    let g = path(3);
    let first = g.edge(10).unwrap_err();
    // An unchecked raw call that fails leaves a record behind...
    let mut from = 0;
    let mut to = 0;
    error::ensure_init();
    let code = unsafe { ffi::igraph_edge(&g, 99, &mut from, &mut to) };
    assert_ne!(code, ffi::igraph_error_type_t_IGRAPH_SUCCESS);
    // ...which must not be attached to a later, different error.
    let second = g.degree(7, NeighborMode::All, Loops::Twice).unwrap_err();
    assert_eq!(second.kind(), ErrorKind::InvalidVertexId);
    assert_ne!(second.message(), first.message());
    // And successful calls still succeed.
    assert!(g.edge(0).is_ok());
}

#[test]
fn check_converts_codes() {
    assert!(check(ffi::igraph_error_type_t_IGRAPH_SUCCESS).is_ok());
    let e = check(ffi::igraph_error_type_t_IGRAPH_EINVAL).unwrap_err();
    assert_eq!(e.kind(), ErrorKind::InvalidValue);
}

#[test]
fn panics_in_callbacks_are_caught_and_resumed() {
    // Simulates a trampoline: the panic is stored, igraph sees INTERRUPTED.
    let code = catch_panic(|| panic!("user callback exploded"));
    assert_eq!(code, ffi::igraph_error_type_t_IGRAPH_INTERRUPTED);
    assert!(error::has_pending_panic());
    // A second panic before the check does not overwrite the first.
    let _ = catch_panic(|| panic!("second"));
    let resumed = std::panic::catch_unwind(|| check(code)).unwrap_err();
    assert_eq!(
        resumed.downcast_ref::<&str>(),
        Some(&"user callback exploded")
    );
    assert!(!error::has_pending_panic());
    // Nothing pending afterwards: checks behave normally again.
    assert!(check(ffi::igraph_error_type_t_IGRAPH_SUCCESS).is_ok());
    assert_eq!(error::catch_panic_or(7, || 3), 3);
    assert_eq!(error::catch_panic_or(7, || -> i32 { panic!("x") }), 7);
    let _ = std::panic::catch_unwind(error::resume_panic);
}

#[test]
fn every_thread_gets_its_own_igraph_state() {
    let handles: Vec<_> = (0..8)
        .map(|t| {
            std::thread::spawn(move || {
                assert!(!error::is_initialized());
                // Errors are reported on the thread that caused them.
                let g = Graph::new(3, false);
                assert!(error::is_initialized());
                let e = g.edge(t).unwrap_err();
                assert_eq!(e.kind(), ErrorKind::InvalidEdgeId);
                // Seeding is per thread and reproducible.
                rng::seed(t as u64).unwrap();
                let a: Vec<i64> = (0..20).map(|_| rng::integer(0, 1000)).collect();
                rng::seed(t as u64).unwrap();
                let b: Vec<i64> = (0..20).map(|_| rng::integer(0, 1000)).collect();
                assert_eq!(a, b);
                let ring = cycle(10 + t);
                ring.degree(.., NeighborMode::All, Loops::Twice)
                    .unwrap()
                    .iter()
                    .sum::<i64>()
            })
        })
        .collect();
    for (t, h) in handles.into_iter().enumerate() {
        assert_eq!(h.join().unwrap(), 2 * (10 + t as i64));
    }
}

#[test]
fn containers_can_move_across_threads_and_be_cloned_there() {
    let v = VectorInt::from([1, 2, 3]);
    let m = Matrix::identity(3);
    let g = karate();
    let sv: StrVector = ["a", "b"].into_iter().collect();
    let h = std::thread::spawn(move || {
        // Clone runs on a fresh thread before any other igraph call.
        let (v2, m2, g2, sv2) = (v.clone(), m.clone(), g.clone(), sv.clone());
        (v2.sum(), m2.sum(), g2.ecount(), sv2.len())
    });
    assert_eq!(h.join().unwrap(), (6, 3.0, 78, 2));
}

#[test]
fn warnings_are_collected() {
    take_warnings();
    let c = std::ffi::CString::new("a test warning").unwrap();
    let f = std::ffi::CString::new("test.c").unwrap();
    error::ensure_init();
    unsafe { ffi::igraph_warning(c.as_ptr(), f.as_ptr(), 42) };
    let w = take_warnings();
    assert_eq!(w, vec!["a test warning (test.c:42)".to_string()]);
    assert!(take_warnings().is_empty());
}

// ---------------------------------------------------------------------------
// Vectors
// ---------------------------------------------------------------------------

#[test]
fn vector_round_trips_and_conversions() {
    let v = Vector::from(vec![1.5, -2.0, 3.25]);
    assert_eq!(Vec::from(&v), vec![1.5, -2.0, 3.25]);
    let w: Vector = v.iter().copied().collect();
    assert_eq!(v, w);
    assert_eq!(v, vec![1.5, -2.0, 3.25]);
    assert_eq!(v.to_string(), "[1.5, -2.0, 3.25]");
    let empty = VectorInt::new();
    assert!(empty.is_empty());
    assert_eq!(empty.min(), None);
    assert_eq!(empty.which_max(), None);
    assert_eq!(empty.sum(), 0);
    let b = VectorBool::from([true, false, true]);
    assert_eq!(b.iter().filter(|&&x| x).count(), 2);
    let v = Vector::view(&[]);
    assert!(v.is_empty());
}

#[test]
fn vector_editing_mirrors_vec() {
    let mut v: VectorInt = (0..6).collect();
    let mut reference: Vec<i64> = (0..6).collect();
    v.insert(2, 100);
    reference.insert(2, 100);
    v.insert(v.len(), 200);
    reference.push(200);
    assert_eq!(v.remove(0), reference.remove(0));
    assert_eq!(v.swap_remove(1), reference.swap_remove(1));
    v.remove_section(1..3);
    reference.drain(1..3);
    v.extend_from_slice(&[7, 8]);
    reference.extend_from_slice(&[7, 8]);
    assert_eq!(v, reference);
    assert_eq!(v.pop(), reference.pop());
    v.truncate(2);
    reference.truncate(2);
    assert_eq!(v, reference);
    v.reserve(100);
    assert!(v.capacity() >= 100);
    v.shrink_to_fit();
    assert_eq!(v.capacity(), v.len());
    v.clear();
    assert!(v.is_empty());
}

#[test]
#[should_panic(expected = "out of bounds")]
fn vector_insert_out_of_bounds_panics() {
    let mut v = Vector::from([1.0]);
    v.insert(3, 0.0);
}

#[test]
fn vector_sorting_and_searching() {
    let mut v = Vector::from([3.0, 1.0, 2.0, 5.0, 4.0]);
    let order = v.sort_ind(Order::Ascending);
    assert_eq!(order, vec![1, 2, 0, 4, 3]);
    let mut sorted = v.clone();
    sorted.permute(&order).unwrap();
    assert_eq!(sorted, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    v.sort();
    assert_eq!(v, sorted);
    assert_eq!(v.binsearch(4.0), Ok(3));
    assert_eq!(v.binsearch(2.5), Err(2));
    assert!(v.contains_sorted(5.0) && !v.contains_sorted(6.0));
    v.reverse_sort();
    assert_eq!(v, vec![5.0, 4.0, 3.0, 2.0, 1.0]);
    assert_eq!(v.sort_ind(Order::Descending), vec![0, 1, 2, 3, 4]);
    assert_eq!(v.search(1, 3.0), Some(2));
    assert_eq!(v.search(3, 3.0), None);

    // min / max family, with ties resolved to the first occurrence.
    let w = VectorInt::from([4, -2, 9, -2, 9]);
    assert_eq!((w.min(), w.max()), (Some(-2), Some(9)));
    assert_eq!(w.which_minmax(), Some((1, 2)));
    assert_eq!(w.minmax(), Some((-2, 9)));

    // Invalid permutations are rejected before reaching igraph.
    let mut x = VectorInt::from([1, 2, 3]);
    assert_eq!(
        x.permute(&[0, 0, 1]).unwrap_err().kind(),
        ErrorKind::InvalidValue
    );
    assert_eq!(
        x.permute(&[0, 3]).unwrap_err().kind(),
        ErrorKind::InvalidValue
    );
    // A shorter permutation keeps only the mentioned elements.
    x.permute(&[2, 0]).unwrap();
    assert_eq!(x, vec![3, 1]);
}

#[test]
fn nan_aware_real_vector_functions() {
    let v = Vector::from([1.0, f64::NAN, -3.0]);
    assert!(v.is_any_nan());
    assert!(!v.is_all_finite());
    assert_eq!(v.is_nan().to_vec(), vec![false, true, false]);
    assert!(v.max().unwrap().is_nan());
    assert!(!v.is_in_interval(-5.0, 5.0));
    // Non-finite values would be undefined behaviour in C: rejected in Rust.
    for bad in [f64::INFINITY, f64::NAN, 1e300] {
        let f = Vector::from([1.5, bad]);
        assert_eq!(f.floor().unwrap_err().kind(), ErrorKind::Overflow);
        assert_eq!(f.round().unwrap_err().kind(), ErrorKind::Overflow);
    }
    let f = Vector::from([1.5, -1.5, 2.49]);
    assert_eq!(f.floor().unwrap(), vec![1, -2, 2]);
    assert_eq!(f.round().unwrap(), vec![2, -2, 2]);
}

#[test]
fn sorted_set_operations() {
    let a = VectorInt::from([1, 2, 2, 3, 5, 8]);
    let b = [2, 3, 4, 8, 9];
    assert_eq!(a.intersect_sorted(&b), vec![2, 3, 8]);
    assert_eq!(a.intersection_size_sorted(&b), 3);
    assert_eq!(a.difference_sorted(&b), vec![1, 2, 5]);
    let (d12, d21, inter) = a.difference_and_intersection_sorted(&b);
    assert_eq!(
        (d12.to_vec(), d21.to_vec(), inter.to_vec()),
        (vec![1, 2, 5], vec![4, 9], vec![2, 3, 8])
    );
    // Multiset identities: |A| = |A \ B| + |A ∩ B| and |B| = |B \ A| + |A ∩ B|.
    assert_eq!(a.len(), d12.len() + inter.len());
    assert_eq!(b.len(), d21.len() + inter.len());
}

#[test]
fn lexicographic_comparisons() {
    let a = VectorInt::from([1, 2, 3]);
    assert_eq!(a.lex_cmp(&[1, 3]), Ordering::Less);
    assert_eq!(a.lex_cmp(&[1, 2]), Ordering::Greater);
    assert_eq!(a.lex_cmp(&[1, 2, 3]), Ordering::Equal);
    // Colexicographic: compare from the end: [1,2,3] vs [3,2,1] -> 3 > 1.
    assert_eq!(a.colex_cmp(&[3, 2, 1]), Ordering::Greater);
    assert!(a.all_l(&[2, 3, 4]) && !a.all_l(&[2, 2, 4]));
    assert!(a.all_le(&[1, 2, 3]) && a.all_ge(&[1, 2, 3]) && !a.all_g(&[1, 2, 3]));
    assert!(!a.all_le(&[1, 2]), "different lengths are never comparable");
    assert_eq!(a.maxdifference(&[1, 5, 3, 1000]), 3.0);
}

#[test]
fn real_vector_arithmetic() {
    let mut v = Vector::from([1.0, 2.0, 3.0, 4.0]);
    assert_eq!(v.sum(), 10.0);
    assert_eq!(v.prod(), 24.0);
    assert_eq!(v.cumsum(), vec![1.0, 3.0, 6.0, 10.0]);
    v.scale(2.0);
    v.add_constant(1.0);
    assert_eq!(v, vec![3.0, 5.0, 7.0, 9.0]);
    v.sub(&[3.0, 5.0, 7.0, 9.0]).unwrap();
    assert!(v.iter().all(|&x| x == 0.0));
    v.add(&[1.0, 1.0, 1.0, 1.0]).unwrap();
    v.mul(&[2.0, 3.0, 4.0, 5.0]).unwrap();
    v.div(&[2.0, 3.0, 4.0, 10.0]).unwrap();
    assert_eq!(v, vec![1.0, 1.0, 1.0, 0.5]);
    assert_eq!(v.add(&[1.0]).unwrap_err().kind(), ErrorKind::InvalidValue);
    let mut z = Vector::from([1.0, 1e-20, -3.0]);
    z.zapsmall(0.0).unwrap();
    assert_eq!(z, vec![1.0, 0.0, -3.0]);
    assert!(z.zapsmall(-1.0).is_err());
    assert!(z.all_almost_e(&[1.0 + 1e-14, 0.0, -3.0], 1e-10));
    let mut a = Vector::from([-1.5, 2.0]);
    a.abs();
    assert_eq!(a, vec![1.5, 2.0]);
    assert_eq!(Vector::range(0.0, 4.0), vec![0.0, 1.0, 2.0, 3.0]);
    assert!(Vector::range(3.0, 1.0).is_empty());
}

#[test]
fn integer_vector_arithmetic_wraps_and_checks() {
    let mut v = VectorInt::from([i64::MAX, 1]);
    assert_eq!(v.sum(), i64::MIN); // wrapping, no C undefined behaviour
    v.add_constant(1);
    assert_eq!(v, vec![i64::MIN, 2]);
    let mut w = VectorInt::from([10, 20, 30]);
    assert_eq!(
        w.div(&[1, 0, 1]).unwrap_err().kind(),
        ErrorKind::InvalidValue
    );
    assert_eq!(
        w,
        vec![10, 20, 30],
        "failed division leaves the vector intact"
    );
    w.div(&[3, 3, 3]).unwrap();
    assert_eq!(w, vec![3, 6, 10]);
    w.mul(&[2, 2, 2]).unwrap();
    w.sub(&[1, 1, 1]).unwrap();
    assert_eq!(w.cumsum(), vec![5, 16, 35]);
    assert_eq!(w.prod(), 5 * 11 * 19);
    let mut n = VectorInt::from([-3, 4, i64::MIN]);
    n.abs();
    assert_eq!(n, vec![3, 4, i64::MIN]);
    assert_eq!(VectorInt::range(2, 6), vec![2, 3, 4, 5]);
    assert!(w.is_in_interval(0, 20) && !w.is_in_interval(6, 20));
    assert!(w.any_smaller(6) && !w.any_smaller(5));
}

#[test]
fn select_shuffle_and_intervals() {
    let v = VectorInt::from([10, 11, 12, 13, 14]);
    assert_eq!(v.select(&[4, 0, 0]).unwrap(), vec![14, 10, 10]);
    assert!(v.select(&[5]).is_err());
    assert_eq!(v.get_interval(1..4), vec![11, 12, 13]);
    let mut m = v.clone();
    m.move_interval(0..2, 3);
    assert_eq!(m, vec![10, 11, 12, 10, 11]);

    rng::seed(3).unwrap();
    let mut a: VectorInt = (0..50).collect();
    a.shuffle();
    rng::seed(3).unwrap();
    let mut b: VectorInt = (0..50).collect();
    b.shuffle();
    assert_eq!(a, b, "shuffles are reproducible under the same seed");
    assert_ne!(a, (0..50).collect::<Vec<_>>());
    a.sort();
    assert_eq!(a, (0..50).collect::<Vec<_>>());
    let mut s = VectorInt::from([1, 2, 2, 2, 2, 5]);
    s.filter_smaller(2);
    assert_eq!(s, vec![2, 2, 5]);
}

#[test]
fn pair_order_sorts_pairs() {
    let first = VectorInt::from([2, 0, 1, 0, 2]);
    let second = [0, 1, 1, 0, 1];
    let order = first.pair_order(&second, 3).unwrap();
    let pairs: Vec<(i64, i64)> = order
        .iter()
        .map(|&i| (first[i as usize], second[i as usize]))
        .collect();
    let mut expected = pairs.clone();
    expected.sort();
    assert_eq!(pairs, expected);
    // `maxval` is the largest value, as in igraph (it sizes the buckets).
    assert_eq!(first.pair_order(&second, 2).unwrap(), order);
    for bad in [1, -1, i64::MAX] {
        assert_eq!(
            first.pair_order(&second, bad).unwrap_err().kind(),
            ErrorKind::InvalidValue
        );
    }
    assert!(first.pair_order(&[0, 1], 3).is_err());
}

#[test]
fn char_and_bool_vectors() {
    let mut c = VectorChar::from_slice(&[5, -3, 7]);
    c.sort();
    assert_eq!(c, vec![-3, 5, 7]);
    c.abs();
    assert_eq!(c.max(), Some(7));
    let mut b = VectorBool::from([true, false]);
    b.insert(1, true);
    assert_eq!(b.search(0, false), Some(2));
    assert_eq!(b.to_string(), "[true, true, false]");
}

#[test]
fn complex_numbers_and_vectors() {
    let z = Complex::new(3.0, 4.0);
    assert_eq!(z.abs(), 5.0);
    assert_eq!(z.conj(), Complex::new(3.0, -4.0));
    assert_eq!(z + Complex::I, Complex::new(3.0, 5.0));
    assert_eq!(z * z.conj(), Complex::new(25.0, 0.0));
    assert!((z / z).almost_equals(Complex::new(1.0, 0.0), 1e-12));
    assert_eq!(-z, Complex::new(-3.0, -4.0));
    let e = Complex::from_polar(1.0, std::f64::consts::PI).add_real(1.0);
    assert!(e.abs() < 1e-12, "Euler: e^(i pi) + 1 = 0");
    assert!(
        Complex::I
            .exp()
            .almost_equals(Complex::new(1f64.cos(), 1f64.sin()), 1e-12)
    );
    assert!(z.ln().exp().almost_equals(z, 1e-12));
    assert!(z.sqrt().powf(2.0).almost_equals(z, 1e-12));
    assert!(Complex::sqrt_real(-4.0).almost_equals(Complex::new(0.0, 2.0), 1e-12));
    let s = z.sin();
    let c = z.cos();
    assert!((s * s + c * c).almost_equals(Complex::new(1.0, 0.0), 1e-9));
    assert!(z.tan().almost_equals(s / c, 1e-12));
    assert!(z.cot().almost_equals(c / s, 1e-12));
    assert!(z.sec().almost_equals(c.inv(), 1e-12));
    assert!(z.csc().almost_equals(s.inv(), 1e-12));
    assert!(
        Complex::new(100.0, 0.0)
            .log10()
            .almost_equals(Complex::new(2.0, 0.0), 1e-12)
    );
    assert!(
        Complex::new(8.0, 0.0)
            .log(Complex::new(2.0, 0.0))
            .almost_equals(Complex::new(3.0, 0.0), 1e-12)
    );
    assert!(
        Complex::I
            .pow(Complex::new(2.0, 0.0))
            .almost_equals(Complex::new(-1.0, 0.0), 1e-12)
    );
    assert_eq!(
        z.mul_real(2.0).sub_imag(8.0).div_real(2.0),
        Complex::new(3.0, 0.0)
    );
    assert_eq!(z.mul_imag(1.0), Complex::new(-4.0, 3.0));
    assert_eq!(
        z.add_imag(1.0).sub_real(3.0).div_imag(5.0),
        Complex::new(1.0, 0.0)
    );
    assert!((z.arg() - (4f64).atan2(3.0)).abs() < 1e-15);
    assert!((z.logabs() - 5f64.ln()).abs() < 1e-15);
    assert_eq!(Complex::new(1.0, 2.0).to_string(), "1+2i");

    let v = VectorComplex::from_parts(&[1.0, 0.0], &[0.0, 1.0]).unwrap();
    assert_eq!(v.sum(), Complex::new(1.0, 1.0));
    assert_eq!(v.prod(), Complex::I);
    let (re, im) = v.realimag();
    assert_eq!((re.to_vec(), im.to_vec()), (vec![1.0, 0.0], vec![0.0, 1.0]));
    assert_eq!(v.real(), re);
    assert_eq!(v.imag(), im);
    let p = VectorComplex::from_polar(&[2.0], &[0.0]).unwrap();
    assert_eq!(p[0], Complex::new(2.0, 0.0));
    let mut w = v.clone();
    w.scale(Complex::new(2.0, 0.0));
    w.add(&v).unwrap();
    assert_eq!(
        w.to_vec(),
        vec![Complex::new(3.0, 0.0), Complex::new(0.0, 3.0)]
    );
    assert_eq!(v.to_string(), "[1+0i, 0+1i]");
    assert!(VectorComplex::from_parts(&[1.0], &[]).is_err());
}

#[test]
fn igraph_style_real_formatting() {
    assert_eq!(format_real(1.0), "1");
    assert_eq!(format_real(-f64::INFINITY), "-Inf");
    assert_close(
        format_real_precise(1.0 / 3.0).parse::<f64>().unwrap(),
        1.0 / 3.0,
        1e-15,
    );
    assert_eq!(format_real_precise(f64::NAN), "NaN");
}

#[test]
fn vectors_survive_drop_and_clone_stress() {
    let mut keep = Vec::new();
    for i in 0..2000 {
        let mut v: VectorInt = (0..i % 37).collect();
        let w = v.clone();
        v.push(i);
        v.insert(0, -1);
        assert_eq!(v.len(), w.len() + 2);
        if i % 100 == 0 {
            keep.push(w);
        }
        let _view = VectorInt::view(&v);
        let _moved = std::mem::take(&mut v);
    }
    assert_eq!(keep.len(), 20);
    assert!(
        keep.iter()
            .all(|v| v.iter().enumerate().all(|(i, &x)| x == i as i64))
    );
}

// ---------------------------------------------------------------------------
// Matrices
// ---------------------------------------------------------------------------

#[test]
fn matrix_construction_and_access() {
    let m = Matrix::from_rows(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).unwrap();
    assert_eq!(m.shape(), (2, 3));
    assert_eq!(m.to_rows(), vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
    assert_eq!(
        m.transposed().to_rows(),
        vec![vec![1.0, 4.0], vec![2.0, 5.0], vec![3.0, 6.0]]
    );
    let mut t = m.clone();
    t.transpose();
    assert_eq!(t, m.transposed());
    assert_eq!(m.diagonal(), vec![1.0, 5.0]);
    let entries: Vec<_> = m.indexed_iter().map(|(ij, &x)| (ij, x)).collect();
    assert_eq!(entries[1], ((1, 0), 4.0));
    assert_eq!(m.get(5, 5), None);
    assert!(Matrix::from_rows(&[vec![1.0], vec![1.0, 2.0]]).is_err());
    assert!(Matrix::from_column_major(2, 2, &[1.0]).is_err());
    let rm = MatrixInt::from_row_major(2, 2, &[1, 2, 3, 4]).unwrap();
    assert_eq!(rm.column(0), &[1, 3]);
    let data = [1.0, 2.0, 3.0, 4.0];
    let view = Matrix::view(&data, 2, 2).unwrap();
    assert_eq!(view[(1, 0)], 2.0);
    assert_eq!(view.sum(), 10.0);
    assert!(Matrix::view(&data, 3, 3).is_err());
}

#[test]
fn matrix_rows_and_columns() {
    let mut m = MatrixInt::from_rows(&[[1, 2], [3, 4], [5, 6]]).unwrap();
    m.set_row(0, &[10, 20]).unwrap();
    m.set_col(1, &[7, 8, 9]).unwrap();
    assert_eq!(m.to_rows(), vec![vec![10, 7], vec![3, 8], vec![5, 9]]);
    assert!(m.set_row(0, &[1]).is_err());
    assert!(m.set_col(5, &[1, 2, 3]).is_err());
    assert_eq!(
        m.select_rows(&[2, 0]).unwrap().to_rows(),
        vec![vec![5, 9], vec![10, 7]]
    );
    assert_eq!(
        m.select_cols(&[1]).unwrap().to_rows(),
        vec![vec![7], vec![8], vec![9]]
    );
    assert_eq!(
        m.select_rows_cols(&[1, 2], &[0]).unwrap().to_rows(),
        vec![vec![3], vec![5]]
    );
    assert!(m.select_rows(&[3]).is_err());
    m.swap_rows(0, 2).unwrap();
    m.swap_cols(0, 1).unwrap();
    assert_eq!(m.to_rows(), vec![vec![9, 5], vec![8, 3], vec![7, 10]]);
    m.remove_row(1).unwrap();
    m.remove_col(0).unwrap();
    assert_eq!(m.to_rows(), vec![vec![5], vec![10]]);
    assert!(m.remove_row(2).is_err());
    m.add_rows(1);
    m.add_cols(2);
    assert_eq!(
        m.to_rows(),
        vec![vec![5, 0, 0], vec![10, 0, 0], vec![0, 0, 0]]
    );
    let mut a = MatrixInt::from_rows(&[[1, 2]]).unwrap();
    a.rbind(&MatrixInt::from_rows(&[[3, 4]]).unwrap()).unwrap();
    a.cbind(&MatrixInt::from_rows(&[[5], [6]]).unwrap())
        .unwrap();
    assert_eq!(a.to_rows(), vec![vec![1, 2, 5], vec![3, 4, 6]]);
    assert!(a.rbind(&MatrixInt::zeros(1, 1)).is_err());
    assert_eq!(a.search(0, 4), Some((1, 1)));
    assert!(a.contains(6) && !a.contains(7));
}

#[test]
fn matrix_arithmetic_and_statistics() {
    let a = Matrix::from_rows(&[[1.0, 2.0], [3.0, 4.0]]).unwrap();
    assert_eq!(a.sum(), 10.0);
    assert_eq!(a.prod(), 24.0);
    assert_eq!(a.rowsums(), vec![3.0, 7.0]);
    assert_eq!(a.colsums(), vec![4.0, 6.0]);
    assert_eq!((a.min(), a.max()), (Some(1.0), Some(4.0)));
    assert_eq!(a.which_max(), Some((1, 1)));
    assert_eq!(a.which_min(), Some((0, 0)));
    assert_eq!(a.minmax(), Some((1.0, 4.0)));
    let mut b = a.clone();
    b.scale(2.0);
    b.add_constant(-1.0);
    assert_eq!(b.to_rows(), vec![vec![1.0, 3.0], vec![5.0, 7.0]]);
    b.sub(&a).unwrap();
    b.mul_elements(&a).unwrap();
    b.div_elements(&a).unwrap();
    assert_eq!(b.to_rows(), vec![vec![0.0, 1.0], vec![2.0, 3.0]]);
    b.add(&Matrix::identity(2)).unwrap();
    assert!(b.add(&Matrix::zeros(3, 3)).is_err());
    assert_eq!(a.maxdifference(&b), 1.0);
    assert!(a.all_ge(&b) && !a.all_g(&b) && b.all_le(&a) && !b.all_l(&a));
    assert!(!a.is_symmetric());
    let s = a.matmul(&a.transposed()).unwrap();
    assert!(s.is_symmetric(), "A·Aᵀ is symmetric");
    assert_eq!(a.mul_vec(&[1.0, 1.0]).unwrap(), a.rowsums());
    assert!(a.matmul(&Matrix::zeros(3, 1)).is_err());
    let mut z = Matrix::from_rows(&[[1.0, 1e-30]]).unwrap();
    z.zapsmall(0.0).unwrap();
    assert_eq!(z.as_slice(), &[1.0, 0.0]);
    assert!(z.all_almost_e(&Matrix::from_rows(&[[1.0, 0.0]]).unwrap(), 1e-12));
    let mut f = Matrix::zeros(2, 2);
    f.fill(0.5);
    assert_eq!(f.sum(), 2.0);
    let e = Matrix::new();
    assert_eq!((e.min(), e.which_max()), (None, None));

    let mut i = MatrixInt::from_rows(&[[1, 2], [3, 4]]).unwrap();
    assert_eq!(
        (i.sum(), i.rowsums(), i.colsums()),
        (10, vec![3, 7], vec![4, 6])
    );
    assert_eq!((i.min(), i.max()), (Some(1), Some(4)));
    i.scale(3);
    i.add_constant(-3);
    i.mul_elements(&MatrixInt::from_rows(&[[1, 1], [1, 0]]).unwrap())
        .unwrap();
    i.sub(&MatrixInt::from_rows(&[[0, 0], [6, 0]]).unwrap())
        .unwrap();
    i.add(&MatrixInt::from_rows(&[[1, 0], [0, 1]]).unwrap())
        .unwrap();
    assert_eq!(i.to_rows(), vec![vec![1, 3], vec![0, 1]]);
    assert_eq!(Matrix::from(&i).sum(), 5.0);
    let bm = MatrixBool::from_rows(&[[true, false], [true, true]]).unwrap();
    assert_eq!(bm.count_true(), 3);
    assert_eq!(
        bm.transposed().to_rows(),
        vec![vec![true, true], vec![false, true]]
    );
}

#[test]
fn matrix_capacity_management() {
    let mut m = Matrix::zeros(10, 10);
    assert!(m.capacity() >= 100);
    m.resize(2, 2);
    m.shrink_to_fit();
    assert_eq!(m.capacity(), 4);
    assert_eq!(m.to_string(), "[0.0, 0.0]\n[0.0, 0.0]\n");
}

// ---------------------------------------------------------------------------
// Lists
// ---------------------------------------------------------------------------

#[test]
fn vector_int_list_editing() {
    let mut l = VectorIntList::from_iter([vec![3, 1], vec![1, 2, 3], vec![1, 2]]);
    l.insert(0, VectorInt::from([9]));
    assert_eq!(l.len(), 4);
    let old = l.replace(0, VectorInt::from([8, 8]));
    assert_eq!(old, vec![9]);
    let removed = l.swap_remove(0);
    assert_eq!(removed, vec![8, 8]);
    assert_eq!(l.to_vecs(), vec![vec![1, 2], vec![3, 1], vec![1, 2, 3]]);
    l[1].push(7);
    l.get_mut(0).unwrap().push(0);
    assert_eq!(l.first().unwrap(), &vec![1, 2, 0]);
    assert_eq!(l.last().unwrap(), &vec![1, 2, 3]);
    l.swap(0, 2);
    l.reverse();
    assert_eq!(
        l.to_vecs(),
        vec![vec![1, 2, 0], vec![3, 1, 7], vec![1, 2, 3]]
    );
    l.push_copy(&VectorInt::from([5]));
    l.extend([VectorInt::from([6])]);
    for v in &mut l {
        v.push(-1);
    }
    assert!(l.iter().all(|v| v.last() == Some(&-1)));
    l.truncate(2);
    assert_eq!(l.len(), 2);
    l.reserve(10);
    assert!(l.capacity() >= 10);
    l.clear();
    assert!(l.is_empty());
    assert_eq!(l.to_string(), "[]");
}

#[test]
fn vector_list_sorting_with_igraph_comparators() {
    let mut l = VectorIntList::from_iter([
        vec![2, 1],
        vec![1, 2, 3],
        vec![1, 2],
        vec![1, 2],
        vec![0, 9],
    ]);
    let ind = l.sort_ind();
    let mut by_ind = l.clone();
    by_ind.permute(&ind).unwrap();
    l.sort();
    assert_eq!(l, by_ind);
    assert_eq!(
        l.to_vecs(),
        vec![
            vec![0, 9],
            vec![1, 2],
            vec![1, 2],
            vec![1, 2, 3],
            vec![2, 1]
        ]
    );
    l.dedup();
    assert_eq!(l.len(), 4);
    l.sort_colex();
    // Colexicographic: compare last elements first.
    assert_eq!(
        l.to_vecs(),
        vec![vec![2, 1], vec![1, 2], vec![1, 2, 3], vec![0, 9]]
    );
    l.sort_by_key(|v| v.len());
    assert_eq!(l[0].len(), 2);
    l.sort_by(|a, b| b.lex_cmp(a));
    assert_eq!(l[0], vec![2, 1]);
    assert!(l.permute(&[0, 0, 1, 2]).is_err());
    assert!(l.permute(&[0, 1]).is_err());
    let mut r = VectorList::from_iter([vec![2.0], vec![1.0, 5.0]]);
    r.sort();
    assert_eq!(r.to_vecs(), vec![vec![1.0, 5.0], vec![2.0]]);
    assert_eq!(r.to_string(), "[[1.0, 5.0], [2.0]]");
}

#[test]
fn graph_and_matrix_lists_own_their_items() {
    let mut graphs = GraphList::new();
    for n in 3..8 {
        graphs.push(cycle(n));
    }
    graphs.insert(0, complete(4));
    graphs.sort_by_key(|g| std::cmp::Reverse(g.ecount()));
    let counts: Vec<usize> = graphs.iter().map(|g| g.ecount()).collect();
    assert_eq!(counts, vec![7, 6, 6, 5, 4, 3]);
    let first = graphs.replace(0, path(2));
    assert_eq!(first.vcount(), 7);
    let copy = graphs.clone();
    graphs.clear();
    assert_eq!(copy.len(), 6);
    graphs.set_directed(true);
    let owned: Vec<Graph> = copy.into_vec();
    assert_eq!(owned[0].ecount(), 1);

    let mut ms = MatrixList::new();
    ms.push(Matrix::identity(2));
    ms.push_copy(&Matrix::identity(3));
    ms.swap(0, 1);
    assert_eq!(ms[0].nrow(), 3);
    ms[1].scale(5.0);
    assert_eq!(ms[1].sum(), 10.0);
}

#[test]
fn lists_survive_stress() {
    for round in 0..200 {
        let mut l = VectorIntList::new();
        for i in 0..20 {
            l.push((0..i).collect());
        }
        if round % 2 == 0 {
            l.reverse();
        }
        let _ = l.remove(3);
        let _ = l.swap_remove(0);
        let c = l.clone();
        let v: Vec<VectorInt> = l.into_iter().collect();
        assert_eq!(v.len(), c.len());
    }
}

// ---------------------------------------------------------------------------
// Bitsets
// ---------------------------------------------------------------------------

#[test]
fn bitset_basics() {
    let mut b = Bitset::new(70); // spans two machine words
    assert_eq!((b.len(), b.count_ones(), b.count_zeros()), (70, 0, 70));
    assert!(b.none() && b.not_all() && !b.any() && !b.all());
    assert_eq!(b.leading_zeros(), 70);
    assert_eq!(b.trailing_zeros(), 70);
    b.set(0, true);
    b.set(64, true);
    b.set(69, true);
    assert!(b.get(64) && !b.get(63));
    assert_eq!(b.count_ones(), 3);
    assert_eq!(b.iter_ones().collect::<Vec<_>>(), vec![0, 64, 69]);
    assert_eq!(b.leading_zeros(), 0);
    assert_eq!(b.trailing_zeros(), 0);
    assert_eq!(b.trailing_ones(), 1);
    assert!(!b.toggle(0));
    assert_eq!(b.trailing_zeros(), 64);
    assert!(b.insert(1) && !b.insert(1));
    b.fill(true);
    assert!(b.all() && b.count_ones() == 70 && b.leading_ones() == 70 && b.trailing_ones() == 70);
    b.clear();
    assert!(b.none());
    b.resize(72);
    assert_eq!(b.len(), 72);
    b.push(true);
    assert_eq!(b.iter_ones().collect::<Vec<_>>(), vec![72]);
    b.reserve(1000);
    assert!(b.capacity() >= 1000);
}

#[test]
fn bitset_algebra_matches_set_algebra() {
    let a = Bitset::from_ones(100, (0..100).filter(|i| i % 2 == 0));
    let b = Bitset::from_ones(100, (0..100).filter(|i| i % 3 == 0));
    let both = &a & &b;
    let either = &a | &b;
    let one = &a ^ &b;
    assert_eq!(
        both.iter_ones().collect::<Vec<_>>(),
        (0..100).filter(|i| i % 6 == 0).collect::<Vec<_>>()
    );
    // Inclusion-exclusion.
    assert_eq!(
        either.count_ones(),
        a.count_ones() + b.count_ones() - both.count_ones()
    );
    assert_eq!(one.count_ones(), either.count_ones() - both.count_ones());
    // De Morgan, and padding bits never leak into the results.
    assert_eq!(!&either, &(!&a) & &(!&b));
    assert_eq!((!&a).count_ones(), 50);
    assert_eq!(a.and(&b), both);
    assert_eq!(a.or(&b), either);
    assert_eq!(a.xor(&b), one);
    let mut c = a.clone();
    c &= &b;
    assert_eq!(c, both);
    c |= &a;
    assert_eq!(c, a);
    c ^= &a;
    assert!(c.none());
    let mut d = Bitset::new(3);
    d.update(&a);
    assert_eq!(d, a);
    let s: Bitset = [true, false, true, true].into_iter().collect();
    assert_eq!(s.to_string(), "1101");
    assert_eq!(format!("{s:>6}"), "  1101");
    assert_eq!(Vec::from(&s), vec![true, false, true, true]);
    assert_eq!(
        Bitset::from(&[false, true][..]).iter().collect::<Vec<_>>(),
        vec![false, true]
    );
    assert!(Bitset::default().is_empty());
}

#[test]
#[should_panic(expected = "different lengths")]
fn bitset_ops_check_lengths() {
    let _ = &Bitset::new(3) | &Bitset::new(4);
}

#[test]
fn bitset_lists() {
    let mut l = BitsetList::new();
    l.push(Bitset::from_ones(5, [1]));
    l.push(Bitset::from_ones(5, [2, 3]));
    l.insert(0, Bitset::new(5));
    assert_eq!(
        l.iter().map(|b| b.count_ones()).collect::<Vec<_>>(),
        vec![0, 1, 2]
    );
    l[0].set(4, true);
    let c = l.clone();
    l.reverse();
    assert_eq!(l[2], c[0]);
    assert_eq!(l.into_vec().len(), 3);
}

// ---------------------------------------------------------------------------
// String vectors
// ---------------------------------------------------------------------------

#[test]
fn strvector_editing() {
    let mut sv = StrVector::with_len(2);
    assert_eq!(sv.to_vec(), vec!["", ""]);
    sv.set(0, "alpha").unwrap();
    sv.set(1, "beta").unwrap();
    assert!(sv.set(2, "x").is_err());
    sv.push("gamma");
    sv.extend(["delta", "epsilon"]);
    assert_eq!(&sv[3], "delta");
    assert_eq!(sv.position("gamma"), Some(2));
    assert!(sv.contains("beta") && !sv.contains("zeta"));
    assert_eq!(sv.remove(1), "beta");
    assert_eq!(sv.pop(), Some("epsilon".to_string()));
    sv.swap(0, 2);
    assert_eq!(sv, *["delta", "gamma", "alpha"].as_slice());
    let picked = sv.select(&[2, 2, 0]).unwrap();
    assert_eq!(picked.to_vec(), vec!["alpha", "alpha", "delta"]);
    assert!(sv.select(&[3]).is_err());
    let mut other: StrVector = vec!["x".to_string(), "y".to_string()].into();
    sv.extend_from(&other);
    assert_eq!(sv.len(), 5);
    sv.append(&mut other);
    assert!(other.is_empty());
    assert_eq!(sv.len(), 7);
    sv.remove_section(1..6);
    assert_eq!(Vec::from(sv.clone()), vec!["delta", "y"]);
    sv.resize(4);
    assert_eq!(sv.to_vec(), vec!["delta", "y", "", ""]);
    sv.truncate(1);
    sv.reserve(50);
    assert!(sv.capacity() >= 50);
    sv.shrink_to_fit();
    assert_eq!(sv.capacity(), 1);
    let all: Vec<String> = (&sv).into_iter().collect();
    assert_eq!(all, vec!["delta"]);
    assert_eq!(sv.to_string(), "[\"delta\"]");
    sv.clear();
    assert!(sv.is_empty());
    // Unicode survives the round trip through C strings.
    sv.push("ĉiuĵaŭde 🦀");
    assert_eq!(sv.get(0), Some("ĉiuĵaŭde 🦀"));
}

// ---------------------------------------------------------------------------
// Selectors
// ---------------------------------------------------------------------------

#[test]
fn vertex_selectors_of_every_kind() {
    let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 0), (2, 3), (3, 3)], 5, false).unwrap();
    assert_eq!(g.select_vertices(..).unwrap(), vec![0, 1, 2, 3, 4]);
    assert_eq!(
        g.select_vertices(VertexSelector::None).unwrap(),
        Vec::<i64>::new()
    );
    assert_eq!(g.select_vertices(3).unwrap(), vec![3]);
    assert_eq!(g.select_vertices(&[4, 0, 4]).unwrap(), vec![4, 0, 4]);
    let owned = vec![1, 2];
    assert_eq!(g.select_vertices(&owned).unwrap(), vec![1, 2]);
    assert_eq!(g.select_vertices(owned.clone()).unwrap(), vec![1, 2]);
    assert_eq!(g.select_vertices(&VectorInt::from([2])).unwrap(), vec![2]);
    assert_eq!(g.select_vertices(1..3).unwrap(), vec![1, 2]);
    assert_eq!(g.select_vertices(1..=3).unwrap(), vec![1, 2, 3]);
    assert_eq!(
        g.select_vertices(VertexSelector::adjacent(2, NeighborMode::All))
            .unwrap(),
        vec![0, 1, 3]
    );
    // A vertex without a self-loop is not adjacent to itself.
    assert_eq!(
        g.select_vertices(VertexSelector::non_adjacent(2, NeighborMode::All))
            .unwrap(),
        vec![2, 4]
    );
    assert_eq!(
        g.select_vertices(VertexSelector::non_adjacent(3, NeighborMode::All))
            .unwrap(),
        vec![0, 1, 4]
    );
    assert_eq!(g.vs_size(..).unwrap(), 5);
    assert_eq!(g.vs_size(&[1, 1, 1]).unwrap(), 3);
    assert_eq!(
        g.select_vertices(&[9]).unwrap_err().kind(),
        ErrorKind::InvalidVertexId
    );
    let raw = VertexSelector::All.to_raw().unwrap();
    assert!(raw.is_all());
    assert_eq!(raw.raw_type(), ffi::igraph_vs_type_t_IGRAPH_VS_ALL);
    let borrowed: &[i64] = &[1, 2];
    let sel = VertexSelector::from(borrowed).into_owned();
    assert_eq!(g.select_vertices(sel).unwrap(), vec![1, 2]);
}

#[test]
fn list_selectors_stay_valid_after_moves() {
    // Regression test: list selectors point to their vector; moving the raw
    // selector around must not leave that pointer dangling.
    let g = complete(6);
    let raws: Vec<_> = (0..50)
        .map(|i| {
            VertexSelector::from(vec![i % 6, (i + 1) % 6])
                .to_raw()
                .unwrap()
        })
        .collect();
    let moved: Vec<_> = raws.into_iter().rev().collect();
    for (k, raw) in moved.iter().enumerate() {
        let i = 49 - k as i64;
        let mut res = VectorInt::new();
        igraph_call!(ffi::igraph_vs_as_vector(&g, raw.get(), &mut res)).unwrap();
        assert_eq!(res, vec![i % 6, (i + 1) % 6]);
    }
    let raws: Vec<_> = (0..15)
        .map(|i| EdgeSelector::from(vec![i, 14 - i]).to_raw().unwrap())
        .collect();
    let moved: Vec<_> = raws.into_iter().collect();
    for (i, raw) in moved.iter().enumerate() {
        let mut res = VectorInt::new();
        igraph_call!(ffi::igraph_es_as_vector(&g, raw.get(), &mut res)).unwrap();
        assert_eq!(res, vec![i as i64, 14 - i as i64]);
    }
}

#[test]
fn edge_selectors_of_every_kind() {
    // 0->1, 1->2, 2->0, 1->2 (multi), 2->3
    let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 0), (1, 2), (2, 3)], 4, true).unwrap();
    assert_eq!(g.select_edges(..).unwrap(), vec![0, 1, 2, 3, 4]);
    let by_from = g
        .select_edges(EdgeSelector::AllOrdered(EdgeOrder::From))
        .unwrap();
    let froms: Vec<i64> = by_from.iter().map(|&e| g.edge_from(e).unwrap()).collect();
    assert!(froms.windows(2).all(|w| w[0] <= w[1]));
    assert!(g.select_edges(EdgeSelector::None).unwrap().is_empty());
    assert_eq!(g.select_edges(3).unwrap(), vec![3]);
    assert_eq!(g.select_edges(&[4, 0]).unwrap(), vec![4, 0]);
    assert_eq!(g.select_edges(1..3).unwrap(), vec![1, 2]);
    assert_eq!(g.select_edges(1..=3).unwrap(), vec![1, 2, 3]);
    let mut out_of_2 = g
        .select_edges(EdgeSelector::incident(2, NeighborMode::Out))
        .unwrap();
    out_of_2.sort();
    assert_eq!(out_of_2, vec![2, 4]);
    let pairs = [(0, 1), (2, 3)];
    assert_eq!(
        g.select_edges(EdgeSelector::pairs(&pairs, true)).unwrap(),
        vec![0, 4]
    );
    // One edge per step; 1 -> 2 is doubled, and either copy may be picked.
    let along = g
        .select_edges(EdgeSelector::path(&[0, 1, 2, 3], true))
        .unwrap();
    assert_eq!((along.len(), along[0], along[2]), (3, 0, 4));
    assert!(along[1] == 1 || along[1] == 3, "{along:?}");
    let mut between = g
        .select_edges(EdgeSelector::all_between(1, 2, true))
        .unwrap();
    between.sort();
    assert_eq!(between, vec![1, 3]);
    assert!(
        g.select_edges(EdgeSelector::all_between(2, 1, true))
            .unwrap()
            .is_empty()
    );
    assert_eq!(
        g.es_size(EdgeSelector::all_between(2, 1, false)).unwrap(),
        2
    );
    assert_eq!(
        g.select_edges(EdgeSelector::pairs(&[(3, 0)], true))
            .unwrap_err()
            .kind(),
        ErrorKind::InvalidValue
    );
    let raw = EdgeSelector::pairs(&pairs, false).to_raw().unwrap();
    assert!(!raw.is_all());
    assert_eq!(raw.raw_type(), ffi::igraph_es_type_t_IGRAPH_ES_PAIRS);
    assert!(EdgeSelector::All.to_raw().unwrap().is_all());
}

// ---------------------------------------------------------------------------
// Graph interface
// ---------------------------------------------------------------------------

#[test]
fn handshake_lemma_on_karate() {
    let g = karate();
    assert_eq!((g.vcount(), g.ecount(), g.is_directed()), (34, 78, false));
    let degrees = g.degree(.., NeighborMode::All, Loops::Twice).unwrap();
    assert_eq!(degrees.iter().sum::<i64>(), 2 * 78);
    // The two leaders of the club are the best connected members.
    let d = VectorInt::from(degrees);
    assert_eq!(d.which_max(), Some(33));
    assert_eq!(d.max(), Some(17));
    assert_eq!(d[0], 16);
    assert_eq!(g.degree_of(0, NeighborMode::All, Loops::Twice).unwrap(), 16);
}

#[test]
fn loops_and_multi_edges_in_neighborhoods() {
    // A loop at 0 and a double edge 0-1.
    let g = Graph::from_edges(&[(0, 0), (0, 1), (0, 1), (1, 2)], 3, false).unwrap();
    assert_eq!(g.neighbors(0, NeighborMode::All).unwrap(), vec![0, 0, 1, 1]);
    assert_eq!(
        g.neighbors_with(0, NeighborMode::All, Loops::Once, true)
            .unwrap(),
        vec![0, 1, 1]
    );
    assert_eq!(
        g.neighbors_with(0, NeighborMode::All, Loops::None, false)
            .unwrap(),
        vec![1]
    );
    assert_eq!(g.degree_of(0, NeighborMode::All, Loops::Twice).unwrap(), 4);
    assert_eq!(g.degree_of(0, NeighborMode::All, Loops::Once).unwrap(), 3);
    assert_eq!(g.degree_of(0, NeighborMode::All, Loops::None).unwrap(), 2);
    assert_eq!(
        g.incident(0, NeighborMode::All, Loops::Twice)
            .unwrap()
            .len(),
        4
    );
    assert_eq!(
        g.incident_edges(0, NeighborMode::All, Loops::Once)
            .unwrap()
            .len(),
        3
    );
    assert_eq!(
        g.adjacent_vertices(0, NeighborMode::All, Loops::None, false)
            .unwrap(),
        vec![1]
    );
    assert_eq!(
        g.adjacent_vertices(0, NeighborMode::All, Loops::Twice, true)
            .unwrap(),
        vec![0, 0, 1, 1]
    );
    let mut between = g.get_all_eids_between(0, 1, false).unwrap();
    between.sort();
    assert_eq!(between, vec![1, 2]);
    assert_eq!(g.other_endpoint(3, 2).unwrap(), 1);
    assert_eq!(g.other_endpoint(0, 0).unwrap(), 0);
    assert!(g.other_endpoint(3, 0).is_err());
}

#[test]
fn directed_graph_interface() {
    let mut g = Graph::new(0, true);
    let a = g.add_vertex().unwrap();
    let b = g.add_vertex().unwrap();
    g.add_vertices(2).unwrap();
    assert_eq!(g.add_edge_id(a, b).unwrap(), 0);
    g.add_edges(&[(1, 2), (2, 3), (3, 1)]).unwrap();
    assert_eq!(g.neighbors(1, NeighborMode::Out).unwrap(), vec![2]);
    assert_eq!(g.neighbors(1, NeighborMode::In).unwrap(), vec![0, 3]);
    assert_eq!(g.neighbors(1, NeighborMode::All).unwrap(), vec![0, 2, 3]);
    assert!(g.has_edge(2, 3, true).unwrap() && !g.has_edge(3, 2, true).unwrap());
    assert!(g.has_edge(3, 2, false).unwrap());
    assert_eq!(g.get_eids(&[(2, 3), (3, 1)], true).unwrap(), vec![2, 3]);
    assert_eq!(
        g.get_eids(&[(3, 2)], true).unwrap_err().kind(),
        ErrorKind::InvalidValue
    );
    assert_eq!(
        g.get_eids_opt(&[(3, 2), (0, 1)], true).unwrap(),
        vec![None, Some(0)]
    );
    assert_eq!(
        g.edges_flat(.., false).unwrap(),
        vec![0, 1, 1, 2, 2, 3, 3, 1]
    );
    assert_eq!(
        g.edges_flat(.., true).unwrap(),
        vec![0, 1, 2, 3, 1, 2, 3, 1]
    );
    assert_eq!(g.edges(&[3]).unwrap(), vec![(3, 1)]);
    assert_eq!((g.edge_from(3).unwrap(), g.edge_to(3).unwrap()), (3, 1));
    let triples: Vec<_> = g.edges_iter().collect();
    assert_eq!(triples[3], (3, 3, 1));
    assert_eq!(g.vertices(), 0..4);
    assert_eq!(g.edge_ids(), 0..4);
    assert_eq!(
        g.degree(.., NeighborMode::In, Loops::Twice).unwrap(),
        vec![0, 2, 1, 1]
    );
    assert_eq!(format!("{g:#}").lines().nth(1), Some("0 -> 1"));
    assert_eq!(g.to_string(), "Directed graph with 4 vertices and 4 edges");
}

#[test]
fn undirected_edges_are_reported_smaller_first() {
    let g = Graph::from_edges(&[(3, 1), (2, 0)], 4, false).unwrap();
    assert_eq!(g.edge_list(), vec![(1, 3), (0, 2)]);
    let via_iter: Vec<_> = g.edges_iter().map(|(_, a, b)| (a, b)).collect();
    assert_eq!(via_iter, g.edge_list());
}

#[test]
fn deleting_vertices_and_edges() {
    let mut g = cycle(6);
    g.delete_edges(&[0, 1]).unwrap();
    assert_eq!(g.ecount(), 4);
    let (map, invmap) = g.delete_vertices_map(&[1, 3]).unwrap();
    assert_eq!(g.vcount(), 4);
    assert_eq!(invmap, vec![0, 2, 4, 5]);
    assert_eq!(map.len(), 6);
    // Surviving vertices map back and forth consistently.
    for (new, &old) in invmap.iter().enumerate() {
        assert_eq!(map[old as usize], new as i64);
    }
    let mut h = complete(5);
    h.delete_vertices(0..2).unwrap();
    assert_eq!((h.vcount(), h.ecount()), (3, 3));
    h.delete_edges(EdgeSelector::incident(0, NeighborMode::All))
        .unwrap();
    assert_eq!(h.ecount(), 1);
    assert_eq!(
        h.delete_vertices(&[7]).unwrap_err().kind(),
        ErrorKind::InvalidVertexId
    );
}

#[test]
fn clone_equality_and_identity() {
    let g = karate();
    let h = g.clone();
    assert_eq!(g, h);
    assert!(g.is_same_graph(&h).unwrap());
    let t = g.try_clone().unwrap();
    assert_eq!(t, g);
    let mut k = g.clone();
    k.add_edge(4, 5).unwrap();
    assert_ne!(g, k);
    // Equality is about the labelled graph: edge order does not matter,
    // but the vertex labels do.
    let a = Graph::from_edges(&[(0, 1), (1, 2)], 3, false).unwrap();
    let b = Graph::from_edges(&[(2, 1), (0, 1)], 3, false).unwrap();
    let c = Graph::from_edges(&[(0, 2), (2, 1)], 3, false).unwrap();
    assert_eq!(a, b);
    assert_ne!(a, c);
    assert_ne!(Graph::new(2, true), Graph::new(2, false));
}

#[test]
fn graph_drop_and_clone_stress() {
    let base = karate();
    let mut graphs = Vec::new();
    for i in 0..300 {
        let mut g = base.clone();
        g.delete_vertices(i % 34).unwrap();
        if i % 3 == 0 {
            graphs.push(g);
        }
    }
    assert!(graphs.iter().all(|g| g.vcount() == 33));
    let total: usize = graphs.iter().map(|g| g.ecount()).sum();
    // Every vertex is deleted equally often among the kept graphs.
    let deg = base.degree(.., NeighborMode::All, Loops::Twice).unwrap();
    let expected: i64 = (0..300).step_by(3).map(|i| 78 - deg[i % 34]).sum();
    assert_eq!(total as i64, expected);
}

#[test]
fn raw_constructors_through_init_with() {
    let star = Graph::init_with(|g| unsafe {
        ffi::igraph_star(g, 6, ffi::igraph_star_mode_t_IGRAPH_STAR_OUT, 0)
    })
    .unwrap();
    assert_eq!(
        star.neighbors(0, NeighborMode::Out).unwrap(),
        vec![1, 2, 3, 4, 5]
    );
    let bad = Graph::init_with(|g| unsafe { ffi::igraph_empty(g, -3, false) });
    assert_eq!(bad.unwrap_err().kind(), ErrorKind::InvalidValue);
}

// ---------------------------------------------------------------------------
// Random numbers
// ---------------------------------------------------------------------------

#[test]
fn rng_generators_are_reproducible_and_independent() {
    for kind in RngType::ALL {
        let mut a = Rng::new(kind, 2024).unwrap();
        let mut b = Rng::new(kind, 2024).unwrap();
        let xs: Vec<i64> = (0..10).map(|_| a.get_integer(-5, 5)).collect();
        let ys: Vec<i64> = (0..10).map(|_| b.get_integer(-5, 5)).collect();
        assert_eq!(xs, ys, "{kind:?}");
        assert!(xs.iter().all(|x| (-5..=5).contains(x)));
        assert!(a.is_seeded());
        assert!(a.bits() >= 31 && a.max() > 0, "{}", a.name());
    }
    assert_eq!(Rng::new(RngType::Mt19937, 1).unwrap().name(), "MT19937");
    assert_eq!(Rng::new(RngType::Pcg64, 1).unwrap().name(), "PCG64");
}

#[test]
fn rng_distributions_have_the_right_moments() {
    let mut r = Rng::new(RngType::Pcg64, 7).unwrap();
    let n = 20_000;
    let mean = |f: &mut dyn FnMut() -> f64| (0..n).map(|_| f()).sum::<f64>() / n as f64;
    assert_close(mean(&mut || r.get_unif01()), 0.5, 0.02);
    assert_close(mean(&mut || r.get_unif(2.0, 4.0)), 3.0, 0.03);
    assert_close(mean(&mut || r.get_normal(10.0, 2.0)), 10.0, 0.1);
    assert_close(mean(&mut || r.get_exp(4.0)), 0.25, 0.02);
    assert_close(mean(&mut || r.get_pois(3.0)), 3.0, 0.1);
    assert_close(mean(&mut || r.get_binom(10, 0.3)), 3.0, 0.1);
    assert_close(mean(&mut || r.get_gamma(2.0, 3.0)), 6.0, 0.2);
    // Geometric: failures before the first success, mean (1 - p) / p.
    assert_close(mean(&mut || r.get_geom(0.25)), 3.0, 0.15);
    assert_close(
        mean(&mut || if r.get_bool() { 1.0 } else { 0.0 }),
        0.5,
        0.02,
    );
}

#[test]
fn default_rng_free_functions() {
    rng::seed(99).unwrap();
    let draws = (
        rng::integer(0, 10),
        rng::uniform(0.0, 1.0),
        rng::uniform01(),
        rng::boolean(),
        rng::normal(0.0, 1.0),
        rng::geometric(0.5),
        rng::binomial(5, 0.5),
        rng::exponential(1.0),
        rng::gamma(1.0, 1.0),
        rng::poisson(1.0),
    );
    rng::seed(99).unwrap();
    let again = (
        rng::integer(0, 10),
        rng::uniform(0.0, 1.0),
        rng::uniform01(),
        rng::boolean(),
        rng::normal(0.0, 1.0),
        rng::geometric(0.5),
        rng::binomial(5, 0.5),
        rng::exponential(1.0),
        rng::gamma(1.0, 1.0),
        rng::poisson(1.0),
    );
    assert_eq!(format!("{draws:?}"), format!("{again:?}"));
    assert_eq!(rng::name(), "PCG32");
    assert!(rng::bits() >= 32 && rng::max() > 0);
}

#[test]
fn scoped_rng_is_restored_even_after_a_panic() {
    rng::seed(5).unwrap();
    let before = rng::name();
    let mut mt = Rng::new(RngType::Mt19937, 1).unwrap();
    let inside = mt.scoped(rng::name);
    assert_eq!(inside, "MT19937");
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        mt.scoped(|| -> () { panic!("inside scoped") })
    }));
    assert!(r.is_err());
    assert_eq!(rng::name(), before);
}

// ---------------------------------------------------------------------------
// A use case: the friendship paradox in Zachary's karate club.
// ---------------------------------------------------------------------------

#[test]
fn story_friendship_paradox_in_the_karate_club() {
    // "Your friends have more friends than you do": on average, the degree
    // of a random neighbour exceeds the average degree, because popular
    // members are counted once for every friend they have.
    let club = karate();
    let degree = VectorInt::from(club.degree(.., NeighborMode::All, Loops::Twice).unwrap());
    let n = club.vcount() as f64;
    let mean_degree = degree.sum() as f64 / n;

    let mut friends_mean = Vector::zeros(club.vcount());
    for v in club.vertices() {
        let friends = club.neighbors(v, NeighborMode::All).unwrap();
        let friend_degrees = degree.select(&friends).unwrap();
        friends_mean[v as usize] = friend_degrees.sum() as f64 / friends.len() as f64;
    }
    let paradox = friends_mean.sum() / n;
    assert!(paradox > mean_degree, "{paradox} <= {mean_degree}");
    assert_close(mean_degree, 156.0 / 34.0, 1e-12);

    // Who experiences it? Members whose friends are more popular than them.
    let mut unlucky = Bitset::new(club.vcount());
    for v in 0..club.vcount() {
        unlucky.set(v, friends_mean[v] > degree[v] as f64);
    }
    // Everyone except a handful of hubs, including both leaders.
    assert!(!unlucky.get(0) && !unlucky.get(33));
    assert!(unlucky.count_ones() >= 28, "{}", unlucky.count_ones());

    // The most "unlucky" member: the largest gap between friends and self.
    let mut gap = friends_mean.clone();
    let as_real: Vec<f64> = degree.iter().map(|&d| d as f64).collect();
    gap.sub(&as_real).unwrap();
    let who = gap.which_max().unwrap();
    // Member 11, whose only friend is the instructor (member 0).
    assert_eq!(who, 11);
    assert_eq!(club.neighbors(11, NeighborMode::All).unwrap(), vec![0]);
    assert_close(gap[who], 15.0, 1e-12);

    // Rank the members by degree, most popular first, and keep a named top 3.
    let ranking = degree.sort_ind(Order::Descending);
    let names: StrVector = ranking
        .iter()
        .take(3)
        .map(|v| format!("member {v}"))
        .collect();
    assert_eq!(names.to_vec(), vec!["member 33", "member 0", "member 32"]);
}

// ---------------------------------------------------------------------------
// Review regressions: sizes, ranges, char / complex matrices.
// ---------------------------------------------------------------------------

#[test]
#[should_panic(expected = "capacity overflow")]
fn huge_bitset_sizes_panic_instead_of_wrapping() {
    // `usize::MAX as i64 == -1`: igraph would accept it and report a length
    // of `usize::MAX` over a one-word buffer.
    let _ = Bitset::new(usize::MAX);
}

#[test]
#[should_panic(expected = "capacity overflow")]
fn huge_vector_sizes_panic_instead_of_aborting() {
    let _ = VectorInt::zeros(usize::MAX);
}

#[test]
#[should_panic(expected = "capacity overflow")]
fn infinite_real_ranges_panic() {
    let _ = Vector::range(0.0, f64::INFINITY);
}

#[test]
#[should_panic(expected = "capacity overflow")]
fn overflowing_integer_ranges_panic() {
    let _ = VectorInt::range(i64::MIN, i64::MAX);
}

#[test]
fn vector_ranges_follow_igraph_lengths() {
    assert_eq!(VectorInt::range(-2, 3), vec![-2, -1, 0, 1, 2]);
    assert!(VectorInt::range(3, -2).is_empty());
    // The length of a real range is `trunc(end - start)`.
    assert_eq!(Vector::range(0.5, 3.0), vec![0.5, 1.5]);
    assert!(Vector::range(0.0, f64::NAN).is_empty());
    assert!(Vector::range(0.0, 0.9).is_empty());
    assert_eq!(VectorChar::range(-1, 2).len(), 3);
}

#[test]
fn zapsmall_uses_an_absolute_tolerance() {
    // A relative tolerance would keep 1e-3 next to 1e6; igraph zaps it.
    let mut v = Vector::from([1e6, 1e-3, -5e-3, 0.02]);
    v.zapsmall(1e-2).unwrap();
    assert_eq!(v, vec![1e6, 0.0, 0.0, 0.02]);
    // With the default tolerance a lonely tiny value is zapped as well.
    let mut w = Vector::from([1e-12]);
    w.zapsmall(0.0).unwrap();
    assert_eq!(w, vec![0.0]);
    assert!(w.zapsmall(-1.0).is_err());
    let mut z = VectorComplex::from_parts(&[1.0, 1e-13], &[1e-13, 2.0]).unwrap();
    z.zapsmall(0.0).unwrap();
    assert_eq!(z.realimag().0, vec![1.0, 0.0]);
    assert_eq!(z.realimag().1, vec![0.0, 2.0]);
}

#[test]
fn complex_vectors_from_parts_and_polar() {
    let z = VectorComplex::from_parts(&[1.0, 0.0], &[0.0, 2.0]).unwrap();
    assert_eq!(z.len(), 2);
    assert_eq!(z[1], Complex::new(0.0, 2.0));
    assert!(VectorComplex::from_parts(&[1.0], &[]).is_err());
    let p = VectorComplex::from_polar(&[2.0], &[std::f64::consts::PI]).unwrap();
    assert!(p[0].almost_equals(Complex::new(-2.0, 0.0), 1e-12));
    // Stress the "initialized by igraph" output path.
    for _ in 0..1000 {
        let _ = VectorComplex::from_parts(&[1.0; 16], &[2.0; 16]).unwrap();
    }
}

#[test]
fn empty_and_decreasing_ranges_select_nothing() {
    let g = Graph::from_edges(&[(0, 1), (1, 2)], 3, false).unwrap();
    // igraph itself rejects `3..3` (start == vcount) and accepts `2..1` with
    // a negative size: both are simply empty in Rust.
    assert!(g.select_vertices(3..3).unwrap().is_empty());
    #[allow(clippy::reversed_empty_ranges)]
    let decreasing = 2..1;
    assert!(g.select_vertices(decreasing.clone()).unwrap().is_empty());
    assert_eq!(g.vs_size(decreasing.clone()).unwrap(), 0);
    assert_eq!(
        g.degree(decreasing, NeighborMode::All, Loops::Twice)
            .unwrap(),
        Vec::<i64>::new()
    );
    #[allow(clippy::reversed_empty_ranges)]
    let inclusive = 5..=3;
    assert!(g.select_vertices(inclusive).unwrap().is_empty());
    assert_eq!(g.es_size(EdgeSelector::Range(1, 0)).unwrap(), 0);
    assert!(g.select_edges(2..2).unwrap().is_empty());
    // Non-empty ranges are still checked against the graph.
    assert!(g.select_vertices(1..4).is_err());
    assert!(g.select_edges(0..=i64::MAX).is_err());
}

#[test]
fn incident_edge_selectors_list_loops_once() {
    let mut g = Graph::from_edges(&[(0, 0), (0, 1), (1, 2)], 3, false).unwrap();
    let mut inc = g
        .select_edges(EdgeSelector::incident(0, NeighborMode::All))
        .unwrap();
    inc.sort();
    assert_eq!(inc, vec![0, 1]);
    // The configurable variant still counts loops twice when asked to.
    let mut twice = g
        .incident_edges(0, NeighborMode::All, Loops::Twice)
        .unwrap();
    twice.sort();
    assert_eq!(twice, vec![0, 0, 1]);
    g.delete_edges(EdgeSelector::incident(0, NeighborMode::All))
        .unwrap();
    assert_eq!(g.edge_list(), vec![(1, 2)]);
}

#[test]
fn delete_vertices_map_marks_deleted_vertices_with_minus_one() {
    let mut g = cycle(6);
    let (map, invmap) = g.delete_vertices_map(&[1, 3]).unwrap();
    assert_eq!(map, vec![0, -1, 1, -1, 2, 3]);
    assert_eq!(invmap, vec![0, 2, 4, 5]);
}

#[test]
fn matrix_shapes_that_overflow_are_rejected() {
    let big = usize::MAX / 2 + 1;
    // `big * 2` wraps around to 0 in release builds: the check must not.
    assert!(Matrix::view(&[], big, 2).is_err());
    assert!(Matrix::from_column_major(big, 2, &[]).is_err());
}

#[test]
fn integer_matrix_division_and_product() {
    let mut a = MatrixInt::from_rows(&[[7, -9], [i64::MIN, 4]]).unwrap();
    assert_eq!(a.prod(), 0); // i64::MIN * 7 * -9 * 4 wraps to 0
    let b = MatrixInt::from_rows(&[[2, 2], [-1, 3]]).unwrap();
    a.div_elements(&b).unwrap();
    assert_eq!(a.to_rows(), vec![vec![3, -4], vec![i64::MIN, 1]]);
    let zero = MatrixInt::zeros(2, 2);
    let before = a.clone();
    assert!(a.div_elements(&zero).is_err());
    assert_eq!(a, before);
    assert_eq!(MatrixInt::new().prod(), 1);
}

#[test]
fn char_matrices() {
    let mut m = MatrixChar::from_rows(&[[1, -2, 3], [4, 5, -6]]).unwrap();
    assert_eq!(m.shape(), (2, 3));
    assert_eq!(m.min(), Some(-6));
    assert_eq!(m.which_max(), Some((1, 1)));
    m.transpose();
    assert_eq!(m.to_rows(), vec![vec![1, 4], vec![-2, 5], vec![3, -6]]);
    m.swap_rows(0, 2).unwrap();
    assert_eq!(m.row(0), vec![3, -6]);
    assert!(m.contains(5));
    assert_eq!(m.to_string(), "[3, -6]\n[-2, 5]\n[1, 4]\n");
    let sel = m.select_cols(&[1]).unwrap();
    assert_eq!(sel.column(0), &[-6, 5, 4]);
}

#[test]
fn complex_matrices() {
    let re = Matrix::from_rows(&[[1.0, 2.0], [3.0, 4.0]]).unwrap();
    let im = Matrix::from_rows(&[[0.5, 0.0], [0.0, -1.0]]).unwrap();
    let mut z = MatrixComplex::from_parts(&re, &im).unwrap();
    assert_eq!(z[(1, 1)], Complex::new(4.0, -1.0));
    assert_eq!(z.real(), re);
    assert_eq!(z.imag(), im);
    assert_eq!(z.sum(), Complex::new(10.0, -0.5));
    assert_eq!(
        z.rowsums(),
        vec![Complex::new(3.0, 0.5), Complex::new(7.0, -1.0)]
    );
    assert_eq!(
        z.colsums(),
        vec![Complex::new(4.0, 0.5), Complex::new(6.0, -1.0)]
    );
    // (1 + 0.5i)(2)(3)(4 - i) = 6 (4 - i + 2i + 0.5) = 27 + 6i
    assert!(z.prod().almost_equals(Complex::new(27.0, 6.0), 1e-12));
    let w = z.clone();
    z.add(&w).unwrap();
    z.scale(Complex::new(0.5, 0.0));
    assert!(z.all_almost_e(&w, 1e-12));
    z.mul_elements(&w).unwrap();
    z.div_elements(&w).unwrap();
    assert!(z.all_almost_e(&w, 1e-12));
    z.sub(&w).unwrap();
    assert_eq!(z.sum(), Complex::new(0.0, 0.0));
    z.add_constant(Complex::I);
    assert_eq!(z[(0, 1)], Complex::I);
    assert!(!z.is_symmetric() || z.shape().0 == z.shape().1);
    assert!(MatrixComplex::from_parts(&re, &Matrix::zeros(1, 2)).is_err());
    let polar = MatrixComplex::from_polar(
        &Matrix::from_rows(&[[1.0]]).unwrap(),
        &Matrix::from_rows(&[[std::f64::consts::FRAC_PI_2]]).unwrap(),
    )
    .unwrap();
    assert!(polar[(0, 0)].almost_equals(Complex::I, 1e-12));
    let mut tiny = MatrixComplex::from_parts(
        &Matrix::from_rows(&[[1e-13, 1.0]]).unwrap(),
        &Matrix::from_rows(&[[1.0, 1e-13]]).unwrap(),
    )
    .unwrap();
    tiny.zapsmall(0.0).unwrap();
    assert_eq!(tiny.as_slice(), &[Complex::I, Complex::new(1.0, 0.0)]);
    assert_eq!(tiny.to_string(), "[0+1i, 1+0i]\n");
}

#[test]
#[should_panic(expected = "empty integer interval")]
fn rng_integer_rejects_reversed_bounds() {
    let _ = rng::integer(5, 3);
}

#[test]
#[should_panic(expected = "invalid real interval")]
fn rng_uniform_rejects_infinite_bounds() {
    let _ = rng::uniform(0.0, f64::INFINITY);
}

#[test]
fn rng_degenerate_intervals() {
    assert_eq!(rng::integer(7, 7), 7);
    assert_eq!(rng::uniform(2.5, 2.5), 2.5);
    let mut r = Rng::new(RngType::Pcg64, 1).unwrap();
    assert_eq!(r.get_integer(i64::MIN, i64::MIN), i64::MIN);
    assert!(rng::poisson(-1.0).is_nan());
    assert!(rng::geometric(0.0).is_nan());
    assert!(rng::exponential(-1.0).is_nan());
}

#[test]
fn init_with_rejects_a_constructor_that_does_nothing() {
    let err = Graph::init_with(|_| ffi::igraph_error_type_t_IGRAPH_SUCCESS).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::Internal);
    let err = Graph::init_with(|g| unsafe { ffi::igraph_empty(g, -1, false) }).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidValue);
    assert!(err.message().contains("negative"));
}

// ---------------------------------------------------------------------------
// igraph 1.0.1 alignment: values pinned by igraph's own C test suite
// (`tests/unit/*.out` of the igraph 1.0.1 sources).
// ---------------------------------------------------------------------------

/// The library loaded at run time must be the one whose headers generated the
/// bindings (and at least 1.0.1, the release these bindings target): a
/// mismatch means the dynamic loader picked another `libigraph.so.4` (e.g.
/// through `LD_LIBRARY_PATH`), and then every other test checks the wrong
/// igraph.
#[test]
fn loaded_igraph_matches_the_headers() {
    let v = igraph::misc::version();
    let headers = (
        igraph::ffi::IGRAPH_VERSION_MAJOR as i32,
        igraph::ffi::IGRAPH_VERSION_MINOR as i32,
        igraph::ffi::IGRAPH_VERSION_PATCH as i32,
    );
    assert!(headers >= (1, 0, 1), "compiled against igraph {headers:?}");
    assert_eq!(
        v.triple(),
        headers,
        "compiled against the igraph {headers:?} headers but loaded igraph {v} at run time"
    );
}

#[test]
fn default_rng_matches_igraph_rng_get_integer_out() {
    // tests/unit/igraph_rng_get_integer.out: the default generator (PCG32).
    let draw = |seed| {
        rng::seed(seed).unwrap();
        (0..10).map(|_| rng::integer(10, 100)).collect::<Vec<_>>()
    };
    let expected = vec![22, 59, 86, 99, 99, 77, 93, 12, 20, 62];
    assert_eq!(draw(42), expected);
    assert_eq!(draw(42), expected);
    assert_eq!(draw(84), vec![33, 48, 87, 63, 80, 14, 15, 95, 79, 43]);
}

/// tests/unit/rng_reproducibility.out: seed 137, 32 integers in `[0, 100]`
/// and then 32 reals in `[0, 1e-6)` (printed with `%g`, 6 digits).
const RNG_137_INTEGERS: [i64; 32] = [
    70, 74, 60, 48, 5, 57, 56, 42, 14, 56, 41, 20, 7, 78, 38, 41, 97, 28, 30, 9, 68, 73, 30, 66,
    11, 41, 22, 41, 4, 99, 35, 71,
];
const RNG_137_REALS: [f64; 32] = [
    4.87906e-07,
    8.04065e-07,
    8.5431e-07,
    3.76015e-07,
    4.00421e-07,
    9.8439e-07,
    3.55795e-08,
    5.80515e-07,
    6.83342e-07,
    3.29069e-07,
    3.99956e-07,
    1.48455e-07,
    5.61698e-07,
    6.7363e-07,
    4.30492e-07,
    1.96356e-07,
    3.13386e-07,
    7.64698e-07,
    5.05553e-07,
    7.90849e-07,
    2.74013e-07,
    8.73595e-08,
    8.06211e-07,
    7.2246e-07,
    2.00342e-07,
    4.3115e-07,
    3.00813e-07,
    4.68466e-07,
    6.67159e-07,
    1.52378e-07,
    7.37829e-07,
    4.32757e-07,
];

fn rng_137_sequence() -> (Vec<i64>, Vec<f64>) {
    rng::seed(137).unwrap();
    let ints = (0..32).map(|_| rng::integer(0, 100)).collect();
    let reals = (0..32).map(|_| rng::uniform(0.0, 1e-6)).collect();
    (ints, reals)
}

fn assert_matches_rng_137((ints, reals): &(Vec<i64>, Vec<f64>)) {
    assert_eq!(ints[..], RNG_137_INTEGERS[..]);
    for (x, e) in reals.iter().zip(RNG_137_REALS) {
        assert!((x - e).abs() <= 1e-5 * e, "{x} != {e}");
    }
}

#[test]
fn default_rng_matches_rng_reproducibility_out() {
    assert_matches_rng_137(&rng_137_sequence());
}

#[test]
fn seeded_results_are_reproducible_in_parallel_threads() {
    // Every thread has its own default generator: seeding in one thread
    // neither disturbs nor is disturbed by the others, so all threads,
    // running at the same time, reproduce igraph's pinned sequence and the
    // same random graph.
    let reference_graph = {
        rng::seed(2024).unwrap();
        Graph::erdos_renyi_game_gnm(30, 60, false, EdgeTypeSw::Simple, false).unwrap()
    };
    let barrier = std::sync::Arc::new(std::sync::Barrier::new(8));
    let handles: Vec<_> = (0..8)
        .map(|_| {
            let barrier = barrier.clone();
            std::thread::spawn(move || {
                barrier.wait();
                let mut runs = Vec::new();
                for _ in 0..20 {
                    runs.push(rng_137_sequence());
                }
                rng::seed(2024).unwrap();
                let g =
                    Graph::erdos_renyi_game_gnm(30, 60, false, EdgeTypeSw::Simple, false).unwrap();
                (runs, g)
            })
        })
        .collect();
    for h in handles {
        let (runs, g) = h.join().unwrap();
        runs.iter().for_each(assert_matches_rng_137);
        assert_eq!(g.ecount(), 60);
        assert_eq!(g, reference_graph);
    }
}

#[test]
fn owned_generators_match_rng_init_destroy_out() {
    // tests/unit/rng_init_destroy_max_bits_name_set_default.out: seed 42,
    // five integers in [0, 100], with each generator and as the default one.
    let expected = [
        (RngType::Glibc2, "LIBC", [3, 69, 20, 64, 30]),
        (RngType::Mt19937, "MT19937", [37, 80, 96, 18, 73]),
        (RngType::Pcg32, "PCG32", [13, 54, 84, 99, 99]),
        (RngType::Pcg64, "PCG64", [61, 38, 73, 84, 67]),
    ];
    for (kind, name, values) in expected {
        let mut r = Rng::new(kind, 7).unwrap();
        assert_eq!(r.name(), name);
        r.set_seed(42).unwrap();
        let draws: Vec<i64> = (0..5).map(|_| r.get_integer(0, 100)).collect();
        assert_eq!(draws, values, "{name}");
        // Installed as the default generator, seeded through the default.
        let scoped = r.scoped(|| {
            assert_eq!(rng::name(), name);
            rng::seed(42).unwrap();
            (0..5).map(|_| rng::integer(0, 100)).collect::<Vec<_>>()
        });
        assert_eq!(scoped, values, "{name} as the default");
        assert!(r.max() >= 0x7fff_ffff && r.bits() >= 31);
    }
    // The thread's own default generator is PCG32 again afterwards.
    rng::seed(42).unwrap();
    let draws: Vec<i64> = (0..5).map(|_| rng::integer(0, 100)).collect();
    assert_eq!(draws, vec![13, 54, 84, 99, 99]);
}

#[test]
fn rng_shuffle_and_vector_shuffle_agree() {
    // Both are Fisher-Yates drawing the same numbers from the same generator.
    let mut slice: Vec<i64> = (0..40).collect();
    rng::seed(11).unwrap();
    rng::shuffle(&mut slice);
    let mut vector: VectorInt = (0..40).collect();
    rng::seed(11).unwrap();
    vector.shuffle();
    assert_eq!(vector, slice);
    assert_ne!(slice, (0..40).collect::<Vec<_>>());
}

#[test]
fn vector_floor_and_round_match_igraph_vector_floor_out() {
    // tests/unit/igraph_vector_floor.out
    let from = Vector::from([-0.6, -0.5, -0.4, -0.0, 0.0, 0.4, 0.5, 0.6, 1.1]);
    assert_eq!(from.floor().unwrap(), vec![-1, -1, -1, 0, 0, 0, 0, 0, 1]);
    // Rounding is half away from zero.
    assert_eq!(from.round().unwrap(), vec![-1, -1, 0, 0, 0, 0, 1, 1, 1]);
}

#[test]
fn list_sorting_matches_igraph_vector_lex_cmp_out() {
    // tests/unit/igraph_vector_lex_cmp.out
    let v = |x: &[f64]| x.iter().map(|y| y * 1e30).collect::<Vec<_>>();
    let mut list = VectorList::from_iter([
        v(&[1.0, 2.0, 9.0]),
        v(&[1.0, 2.0, 3.0]),
        v(&[1.0, 2.0]),
        v(&[]),
        v(&[1.0, 2.0, 9.0]),
        v(&[]),
        v(&[9.0, 2.0, 1.0]),
        v(&[3.0, 3.0]),
    ]);
    list.sort();
    assert_eq!(
        list.to_vecs(),
        vec![
            v(&[]),
            v(&[]),
            v(&[1.0, 2.0]),
            v(&[1.0, 2.0, 3.0]),
            v(&[1.0, 2.0, 9.0]),
            v(&[1.0, 2.0, 9.0]),
            v(&[3.0, 3.0]),
            v(&[9.0, 2.0, 1.0]),
        ]
    );
    list.sort_colex();
    assert_eq!(
        list.to_vecs(),
        vec![
            v(&[]),
            v(&[]),
            v(&[9.0, 2.0, 1.0]),
            v(&[1.0, 2.0]),
            v(&[1.0, 2.0, 3.0]),
            v(&[3.0, 3.0]),
            v(&[1.0, 2.0, 9.0]),
            v(&[1.0, 2.0, 9.0]),
        ]
    );
    // The same orders, one comparison at a time.
    let a = Vector::from(v(&[1.0, 2.0]));
    assert_eq!(a.lex_cmp(&v(&[1.0, 2.0, 3.0])), Ordering::Less);
    assert_eq!(a.colex_cmp(&v(&[9.0, 2.0, 1.0])), Ordering::Greater);
}

#[test]
fn bitset_matches_igraph_bitset_out() {
    // tests/unit/bitset.out, "Test IGRAPH_BIT_SET/CLEAR" and "Test printing".
    let mut b = Bitset::new(35);
    for i in [0, 24, 13, 17, 34, 13] {
        b.set(i, true);
    }
    for i in [33, 34, 17] {
        b.set(i, false);
    }
    assert_eq!(b.iter_ones().collect::<Vec<_>>(), vec![0, 13, 24]);
    assert_eq!(b.to_string(), "00000000001000000000010000000000001");
    assert_eq!(b.clone(), b);
    // `complement` / `!` and `from_bools` agree with the definitions.
    let c = b.complement();
    assert_eq!(c.count_ones(), 32);
    assert_eq!(c, !&b);
    assert_eq!((&b | &c).count_ones(), 35);
    let bools = b.to_vec();
    assert_eq!(Bitset::from_bools(&bools), b);
    assert_eq!(Bitset::from_bools(&[true, false, true]).to_string(), "101");
}

// ---------------------------------------------------------------------------
// Cross-checks against the other modules of the crate.
// ---------------------------------------------------------------------------

#[test]
fn test_helpers_match_igraph_constructors() {
    // The hand-written helpers of `tests/common` are the very labelled graphs
    // built by igraph's constructors.
    assert_eq!(karate(), Graph::famous("Zachary").unwrap());
    for n in 3..8 {
        assert_eq!(
            cycle(n),
            Graph::ring(n as usize, false, false, true).unwrap()
        );
        assert_eq!(
            path(n),
            Graph::ring(n as usize, false, false, false).unwrap()
        );
        assert_eq!(complete(n), Graph::full(n as usize, false, false).unwrap());
    }
}

#[test]
fn degrees_agree_with_strength_adjacency_and_mean_degree() {
    let club = Graph::famous("Zachary").unwrap();
    let degree = club.degree(.., NeighborMode::All, Loops::Twice).unwrap();
    // Unweighted strength is the degree.
    let strength = club
        .strength(.., NeighborMode::All, Loops::Twice, None)
        .unwrap();
    assert_eq!(
        strength,
        degree.iter().map(|&d| d as f64).collect::<Vec<_>>()
    );
    // Row sums of the adjacency matrix are the degrees.
    let adj = club
        .get_adjacency(GetAdjacency::Both, None, Loops::Twice)
        .unwrap();
    assert_eq!(
        adj.rowsums(),
        degree.iter().map(|&d| d as f64).collect::<Vec<_>>()
    );
    assert_eq!(
        club.maxdegree(.., NeighborMode::All, Loops::Twice).unwrap(),
        17
    );
    assert_close(club.mean_degree(true).unwrap(), 156.0 / 34.0, 1e-12);
    // `degree_of` agrees with `degree` for every vertex.
    for v in club.vertices() {
        assert_eq!(
            club.degree_of(v, NeighborMode::All, Loops::Twice).unwrap(),
            degree[v as usize]
        );
    }
}

#[test]
fn degree_of_validates_the_vertex() {
    // Regression: igraph_degree_1 (igraph 1.0.0 and 1.0.1) does not check
    // the vertex id and would read out of bounds.
    let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 2)], 3, true).unwrap();
    for v in [3, 7, -1, i64::MAX, i64::MIN] {
        for mode in [NeighborMode::Out, NeighborMode::In, NeighborMode::All] {
            let err = g.degree_of(v, mode, Loops::Twice).unwrap_err();
            assert_eq!(err.kind(), ErrorKind::InvalidVertexId, "{v}");
        }
    }
    // And on valid vertices it matches `degree` in every mode.
    for mode in [NeighborMode::Out, NeighborMode::In, NeighborMode::All] {
        for loops in [Loops::None, Loops::Once, Loops::Twice] {
            let all = g.degree(.., mode, loops).unwrap();
            let one: Vec<i64> = g
                .vertices()
                .map(|v| g.degree_of(v, mode, loops).unwrap())
                .collect();
            assert_eq!(one, all, "{mode:?} {loops:?}");
        }
    }
    assert_eq!(
        g.degree(2, NeighborMode::All, Loops::Twice).unwrap(),
        vec![3]
    );
    assert_eq!(
        g.degree(2, NeighborMode::All, Loops::Once).unwrap(),
        vec![2]
    );
    assert_eq!(
        g.degree(2, NeighborMode::Out, Loops::Once).unwrap(),
        vec![1],
        "a directed loop counts once towards the out-degree anyway"
    );
}

#[test]
fn selector_sizes_only_count() {
    // igraph_vs_size / igraph_es_size count without validating lists,
    // ranges and single ids; selectors that query the graph do validate.
    let g = Graph::ring(5, false, false, true).unwrap();
    assert_eq!(g.vs_size(&[9, 9]).unwrap(), 2);
    assert_eq!(g.vs_size(9).unwrap(), 0);
    assert_eq!(g.vs_size(3).unwrap(), 1);
    assert_eq!(g.vs_size(0..9).unwrap(), 9);
    assert_eq!(
        g.vs_size(VertexSelector::adjacent(9, NeighborMode::All))
            .unwrap_err()
            .kind(),
        ErrorKind::InvalidVertexId
    );
    assert_eq!(g.es_size(&[42]).unwrap(), 1);
    assert_eq!(g.es_size(42).unwrap(), 0);
    assert_eq!(
        g.es_size(EdgeSelector::incident(9, NeighborMode::All))
            .unwrap_err()
            .kind(),
        ErrorKind::InvalidVertexId
    );
    assert_eq!(
        g.es_size(EdgeSelector::pairs(&[(0, 2)], false))
            .unwrap_err()
            .kind(),
        ErrorKind::InvalidValue
    );
    // Resolving validates.
    assert!(g.select_vertices(&[9]).is_err());
    assert!(g.select_edges(&[42]).is_err());
    // Adjacency-based functions agree with their selector counterparts.
    let h = Graph::from_edges(&[(0, 0), (0, 1), (0, 1), (1, 2)], 3, false).unwrap();
    for loops in [Loops::None, Loops::Once, Loops::Twice] {
        for multiple in [false, true] {
            assert_eq!(
                h.adjacent_vertices(0, NeighborMode::All, loops, multiple)
                    .unwrap(),
                h.neighbors_with(0, NeighborMode::All, loops, multiple)
                    .unwrap()
            );
        }
        assert_eq!(
            h.incident_edges(0, NeighborMode::All, loops).unwrap().len(),
            h.incident(0, NeighborMode::All, loops).unwrap().len()
        );
    }
}

#[test]
fn remaining_public_helpers() {
    // Errors built on the Rust side.
    let e = Error::invalid("bad input");
    assert_eq!(e.kind(), ErrorKind::InvalidValue);
    assert_eq!((e.message(), e.file(), e.line()), ("bad input", "", 0));
    assert_eq!(
        e.to_string(),
        format!("{}: bad input", ErrorKind::InvalidValue.description())
    );

    // Graph basics.
    Graph::setup();
    assert!(error::is_initialized());
    let mut g = Graph::empty(3, false).unwrap();
    assert_eq!((g.num_vertices(), g.num_edges()), (3, 0));
    g.add_edges_from_slice(&[(0, 1)]).unwrap();
    g.add_edges_from_vector(&[1, 2, 2, 0]).unwrap();
    assert!(g.add_edges_from_vector(&[0]).is_err());
    assert_eq!(g.edge_list(), vec![(0, 1), (1, 2), (0, 2)]);
    assert_eq!(g.get_eid(2, 1, true).unwrap(), Some(1));
    assert_eq!(g.get_eid(0, 0, true).unwrap(), None);
    assert_eq!(
        g.get_eid(0, 5, true).unwrap_err().kind(),
        ErrorKind::InvalidVertexId
    );
    g.invalidate_cache();
    assert_eq!(g, Graph::ring(3, false, false, true).unwrap());

    // Mutable access to the raw storage of containers.
    let mut v = VectorInt::from([3, 1, 2]);
    assert_eq!(v.size(), 3);
    v.as_mut_slice().sort();
    assert_eq!(v, vec![1, 2, 3]);
    #[allow(deprecated)]
    let zeros = Vector::with_capacity(2);
    assert_eq!(zeros, vec![0.0, 0.0]);
    let z = Complex::new(1.5, -2.0);
    assert_eq!((z.re(), z.im()), (1.5, -2.0));

    let mut m = Matrix::from_rows(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).unwrap();
    m.as_mut_slice()[0] = 10.0; // column-major: element (0, 0)
    assert_eq!(m.ncol(), 3);
    assert_eq!(
        m.rows().collect::<Vec<_>>(),
        vec![vec![10.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]
    );
    assert_eq!(
        m.columns().collect::<Vec<_>>(),
        vec![&[10.0, 4.0][..], &[2.0, 5.0][..], &[3.0, 6.0][..]]
    );

    let mut l = VectorIntList::from_iter([vec![2, 1], vec![3]]);
    for item in l.iter_mut() {
        item.push(0);
    }
    l.as_mut_slice().swap(0, 1);
    assert_eq!(l.to_vecs(), vec![vec![3, 0], vec![2, 1, 0]]);

    let sv: StrVector = ["alpha", "beta"].into_iter().collect();
    assert_eq!(sv.get_cstr(1).unwrap().to_bytes(), b"beta");
    assert_eq!(sv.get_cstr(2), None);
}

// ---------------------------------------------------------------------------
// igraph calls nested inside callbacks (finally-stack levels and depth)
// ---------------------------------------------------------------------------

mod nested_callbacks {
    use super::*;
    use igraph::error::{catch_panic_or, finally_stack_size};
    use igraph::visitor::BfsOptions;
    use std::ffi::c_void;
    use std::ops::ControlFlow;
    use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

    /// A failing igraph call made by a visitor frees only its own
    /// temporaries: the outer BFS keeps its queue and result vectors (before
    /// the fix, the error handler freed them while the search was running).
    #[test]
    fn failing_call_inside_bfs_visitor_does_not_free_the_outer_search() {
        let g = karate();
        let expected = g
            .bfs_with(&[0], &BfsOptions::default(), |_| ControlFlow::Continue(()))
            .unwrap();
        let probe = path(3);
        let mut sizes = Vec::new();
        let res = g
            .bfs_with(&[0], &BfsOptions::default(), |_| {
                let before = finally_stack_size();
                // Several failing calls per visit, of different kinds.
                let e1 = probe.neighbors(99, NeighborMode::All).unwrap_err();
                assert_eq!(e1.kind(), ErrorKind::InvalidVertexId);
                let e2 = Graph::from_edges(&[(0, -1)], 2, false).unwrap_err();
                assert_eq!(e2.kind(), ErrorKind::InvalidVertexId);
                let mut h = probe.clone();
                assert!(h.delete_edges(&[7][..]).is_err());
                // The outer search's entries are all still there.
                assert_eq!(finally_stack_size(), before);
                sizes.push(before);
                ControlFlow::Continue(())
            })
            .unwrap();
        assert_eq!(res.order, expected.order);
        assert_eq!(res.dist, expected.dist);
        assert_eq!(res.parents, expected.parents);
        assert_eq!(sizes.len(), 34);
        // The BFS keeps temporaries on the stack while it runs the visitor.
        assert!(sizes.iter().all(|&s| s > 0));
        assert_eq!(finally_stack_size(), 0);
        // The thread's error machinery still works normally afterwards.
        assert_eq!(
            probe.neighbors(5, NeighborMode::All).unwrap_err().kind(),
            ErrorKind::InvalidVertexId
        );
    }

    struct RawBfs {
        probe: Graph,
        visited: Vec<i64>,
        sizes_ok: bool,
    }

    unsafe extern "C" fn raw_bfs_handler(
        _graph: *const ffi::igraph_t,
        vid: ffi::igraph_int_t,
        _pred: ffi::igraph_int_t,
        _succ: ffi::igraph_int_t,
        _rank: ffi::igraph_int_t,
        _dist: ffi::igraph_int_t,
        extra: *mut c_void,
    ) -> ffi::igraph_error_t {
        catch_panic(|| {
            let st = unsafe { &mut *(extra as *mut RawBfs) };
            let before = finally_stack_size();
            let failed = st.probe.neighbors(-3, NeighborMode::All);
            st.sizes_ok &= failed.is_err() && finally_stack_size() == before;
            st.visited.push(vid);
            ffi::igraph_error_type_t_IGRAPH_SUCCESS
        })
    }

    /// The same through the raw `igraph_bfs` and a hand-written trampoline
    /// that only uses the core `catch_panic`.
    #[test]
    fn failing_call_inside_raw_bfs_callback() {
        let g = cycle(6);
        let mut st = RawBfs {
            probe: path(2),
            visited: Vec::new(),
            sizes_ok: true,
        };
        let mut order = VectorInt::new();
        igraph_call!(ffi::igraph_bfs(
            &g,
            0,
            std::ptr::null(),
            ffi::igraph_neimode_t_IGRAPH_ALL,
            false,
            std::ptr::null(),
            &mut order,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            Some(raw_bfs_handler),
            &mut st as *mut RawBfs as *mut c_void,
        ))
        .unwrap();
        assert!(st.sizes_ok);
        assert_eq!(st.visited, order.to_vec());
        assert_eq!(st.visited.len(), 6);
        assert_eq!(finally_stack_size(), 0);
    }

    /// Runs a BFS on a 4-cycle whose visitor, at the root, starts the same
    /// search again, `levels` times in total. Returns the number of levels
    /// that ran, or the first error.
    fn nested_bfs(g: &Graph, level: usize, levels: usize) -> Result<usize> {
        let mut inner: Result<usize> = Ok(level);
        g.bfs_with(&[0], &BfsOptions::default(), |v| {
            if v.vid == 0 && level + 1 < levels {
                inner = nested_bfs(g, level + 1, levels);
                if inner.is_err() {
                    return ControlFlow::Break(());
                }
            }
            ControlFlow::Continue(())
        })?;
        inner
    }

    #[test]
    fn moderately_nested_searches_work() {
        let g = cycle(4);
        assert_eq!(nested_bfs(&g, 0, 8).unwrap(), 7);
        assert_eq!(finally_stack_size(), 0);
    }

    /// igraph's finally stack has 100 entries and igraph aborts the process
    /// when it overflows: deeply nested calls are refused with an error.
    #[test]
    fn too_deeply_nested_searches_fail_instead_of_aborting() {
        let g = cycle(4);
        let err = nested_bfs(&g, 0, 200).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::Failure);
        assert!(err.message().contains("nested too deeply"), "{err}");
        assert_eq!(finally_stack_size(), 0);
        // Nothing is left behind: the thread can nest again.
        assert_eq!(nested_bfs(&g, 0, 5).unwrap(), 4);
    }

    #[test]
    fn panic_in_a_nested_callback_unwinds_all_levels_cleanly() {
        let g = cycle(5);
        let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            g.bfs_with(&[0], &BfsOptions::default(), |_| {
                let _ = g.bfs_with(&[1], &BfsOptions::default(), |w| {
                    if w.vid == 3 {
                        panic!("inner visitor gave up");
                    }
                    ControlFlow::Continue(())
                });
                ControlFlow::Continue(())
            })
        }));
        let payload = caught.unwrap_err();
        assert_eq!(
            payload.downcast_ref::<&str>(),
            Some(&"inner visitor gave up")
        );
        assert!(!error::has_pending_panic());
        assert_eq!(finally_stack_size(), 0);
        // Errors and searches keep working on this thread.
        assert!(g.neighbors(9, NeighborMode::All).is_err());
        assert_eq!(nested_bfs(&g, 0, 3).unwrap(), 2);
    }

    static NOOP_DESTRUCTOR_RUNS: AtomicUsize = AtomicUsize::new(0);

    unsafe extern "C" fn counting_destructor(_ptr: *mut c_void) {
        NOOP_DESTRUCTOR_RUNS.fetch_add(1, AtomicOrdering::SeqCst);
    }

    /// Entries left on the stack by the callback (as a C function returning
    /// an error without reporting it would do) are dropped when the callback
    /// returns, without running their destructors, and the level is closed.
    #[test]
    fn entries_leaked_inside_a_callback_are_dropped_without_destructors() {
        let before = NOOP_DESTRUCTOR_RUNS.load(AtomicOrdering::SeqCst);
        let v = catch_panic_or(false, || {
            unsafe {
                ffi::IGRAPH_FINALLY_REAL(Some(counting_destructor), std::ptr::null_mut());
                ffi::IGRAPH_FINALLY_REAL(Some(counting_destructor), std::ptr::null_mut());
            }
            assert_eq!(finally_stack_size(), 2);
            true
        });
        assert!(v);
        assert_eq!(finally_stack_size(), 0);
        assert_eq!(NOOP_DESTRUCTOR_RUNS.load(AtomicOrdering::SeqCst), before);
        // The level was closed: an error now frees an empty stack, normally.
        assert!(path(2).neighbors(4, NeighborMode::All).is_err());
        assert_eq!(finally_stack_size(), 0);
    }

    #[test]
    fn catch_panic_or_returns_the_fallback_and_closes_the_level_on_panic() {
        let v = catch_panic_or(-1, || -> i32 { panic!("boom") });
        assert_eq!(v, -1);
        assert!(error::has_pending_panic());
        assert_eq!(finally_stack_size(), 0);
        let resumed = std::panic::catch_unwind(error::resume_panic).unwrap_err();
        assert_eq!(resumed.downcast_ref::<&str>(), Some(&"boom"));
    }
}

#[test]
fn counts_that_do_not_fit_igraph_int_are_rejected() {
    let big = usize::MAX;
    assert_eq!(
        Graph::empty(big, false).unwrap_err().kind(),
        ErrorKind::InvalidValue
    );
    assert_eq!(
        Graph::from_flat_edges(&[0, 1], big, true)
            .unwrap_err()
            .kind(),
        ErrorKind::InvalidValue
    );
    assert_eq!(
        Graph::from_edges(&[(0, 1)], (i64::MAX as usize) + 1, true)
            .unwrap_err()
            .kind(),
        ErrorKind::InvalidValue
    );
    let mut g = Graph::new(2, false);
    let err = g.add_vertices(big).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidValue);
    assert!(err.message().contains("does not fit"));
    assert_eq!(g.vcount(), 2);
    // Counts that fit but exceed igraph's limits are igraph's errors.
    assert_eq!(
        Graph::empty(i64::MAX as usize, false).unwrap_err().kind(),
        ErrorKind::Range
    );
    let err = g.add_vertices(i64::MAX as usize).unwrap_err();
    assert!(matches!(err.kind(), ErrorKind::Overflow | ErrorKind::Range));
    assert_eq!(g.vcount(), 2);
    g.add_vertices(3).unwrap();
    assert_eq!(g.vcount(), 5);
}

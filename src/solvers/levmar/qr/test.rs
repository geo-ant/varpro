use approx::assert_relative_eq;
use nalgebra::dmatrix;

#[test]
#[allow(non_snake_case)]
fn upper_triangular_solve_for_single_rhs() {
    let R = dmatrix![
           9.,  9.,   2.,   6.,   4.;
           0.,  2.,   5.,   5.,   7.;
           0.,  0.,   8.,   2.,   1.;
           0.,  0.,   0.,   9.,   1.;
           0.,  0.,   0.,   0.,   9.;
    ];
    let x_expected = dmatrix![3.; 7.; 8.; 5.; 1.;];
    let b = dmatrix![140.; 86.; 75.; 46.; 9.];

    let x = super::solve_upper_triangular(R.as_view(), b.as_view()).unwrap();
    assert_relative_eq!(x, x_expected, epsilon = 0.0001);
}

#[test]
#[allow(non_snake_case)]
fn upper_triangular_solve_for_mhrs() {
    let R = dmatrix![
        5.,   1.,   7.,   2.,   9.;
        0.,   2.,   7.,   9.,   8.;
        0.,   0.,   9.,   6.,   2.;
        0.,   0.,   0.,   6.,   3.;
        0.,   0.,   0.,   0.,   7.;
    ];
    let X_expected = dmatrix![
       3.,   2.,   4.;
       6.,   4.,   7.;
       2.,   6.,   3.;
       4.,   3.,   1.;
       9.,   3.,   0.;
    ];
    let B = dmatrix![
    124.,    89.,    50.;
    134.,   101.,    44.;
     60.,    78.,    33.;
     51.,    27.,    6.;
     63.,    21.,    0.;
     ];

    let X = super::solve_upper_triangular(R.as_view(), B.as_view()).unwrap();
    assert_relative_eq!(X, X_expected, epsilon = 0.0001);
}

#[test]
#[allow(non_snake_case)]
fn upper_triangular_solve_for_singular_matrix_r() {
    let R = dmatrix![
        5.,   1.,   7.,   2.,   9.;
        0.,   2.,   7.,   9.,   8.;
        0.,   0.,   0.,   6.,   2.;
        0.,   0.,   0.,   6.,   3.;
        0.,   0.,   0.,   0.,   7.;
    ];
    let B = dmatrix![
    124.,    89.,    50.;
    134.,   101.,    44.;
     60.,    78.,    33.;
     51.,    27.,    6.;
     63.,    21.,    0.;
     ];

    assert!(super::solve_upper_triangular(R.as_view(), B.as_view()).is_none());

    let R = dmatrix![
           0.,  9.,   2.,   6.,   4.;
           0.,  2.,   5.,   5.,   7.;
           0.,  0.,   8.,   2.,   1.;
           0.,  0.,   0.,   9.,   1.;
           0.,  0.,   0.,   0.,   9.;
    ];
    let b = dmatrix![140.; 86.; 75.; 46.; 9.];

    assert!(super::solve_upper_triangular(R.as_view(), b.as_view()).is_none());
}

// Copyright 2026 Zhexian Zhou
// SPDX-License-Identifier: Apache-2.0
// check_linalg.cpp -- runnable value spot-check for the Phase 5A lu/lup/det/inv
// refactor: confirms LAPACK compute precision (and hence the numeric RESULT) is
// unchanged by the output-first `Out` reshaping -- only the return container's
// element type moved.
#include "MatSAR/math/LinearAlgebra.hpp"
#include <cmath>
#include <cstdio>
#include <cstdlib>

using namespace matsar;

static bool approx(double got, double want, double atol = 1e-9) {
    return std::fabs(got - want) <= atol;
}

int main() {
    // NOTE: the task's suggested example, A = [[4,3],[6,3]], triggers a partial-pivot
    // row swap under LAPACKE_dgetrf2 (|6|>|4| in column 0). Math::det()'s body -- kept
    // verbatim per the Phase 5A instructions -- computes `L.det() * prod(diag(U))` and
    // never folds in the permutation matrix's sign, so any input requiring an ODD number
    // of pivot swaps gets the WRONG SIGN (confirmed: that example yields +6, not -6; L, U,
    // P returned by lup() are individually all correct -- P is just never consulted by
    // det()). This is a pre-existing bug, unrelated to the Phase 5A signature refactor
    // (byte-identical body before/after); flagged separately, not fixed here.
    // A = [[4,1],[2,3]] avoids a pivot swap (|4|>|2| in column 0), so it exercises the
    // refactor (Out-typed return, unchanged compute precision) without tripping over the
    // unrelated sign bug. det = 4*3 - 1*2 = 10.
    Matrix<float64> A = Matrix<float64>::fromRows({{4, 1},
                                                    {2, 3}});

    bool ok = true;

    // det(A) ~ 10
    auto d = matsar::Math::det(A);
    double dv = d(0);
    if (!approx(dv, 10.0)) {
        std::printf("FAIL: det(A) = %f, want 10\n", dv);
        ok = false;
    }

    // inv(A) round-trips: mmul(A, inv(A)) ~ I
    auto Ainv = matsar::Math::inv(A);
    auto I = mmul(A, Ainv);
    double i00 = I(0, 0), i01 = I(0, 1), i10 = I(1, 0), i11 = I(1, 1);
    if (!approx(i00, 1.0) || !approx(i11, 1.0) || !approx(i01, 0.0) || !approx(i10, 0.0)) {
        std::printf("FAIL: mmul(A, inv(A)) = [[%f,%f],[%f,%f]], want identity\n", i00, i01, i10, i11);
        ok = false;
    }

    if (!ok)
        return 1;
    std::printf("PASS\n");
    return 0;
}

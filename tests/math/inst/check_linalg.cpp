// Copyright 2026 Zhexian Zhou
// SPDX-License-Identifier: Apache-2.0
// check_linalg.cpp -- runnable value spot-check for the lu/lup/det/inv refactor and
// the three pre-existing correctness fixes it surfaced:
//   1. Matrix::Det() compares against a typed zero (complex Det instantiates).
//   2. det() folds in the LU row-permutation SIGN (correct determinant sign).
//   3. rank() takes the magnitude of the singular values (supports complex;
//      unblocks inv()/pinv() on complex matrices).
#include "MatSAR/math/LinearAlgebra.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>

using namespace matsar;

static bool approx(double got, double want, double atol = 1e-9) {
    return std::fabs(got - want) <= atol;
}

int main() {
    bool ok = true;

    // ---- det SIGN regression guard (the permutation-sign fix) ----------------
    // [[4,3],[6,3]] needs an ODD pivot swap under LAPACK partial pivoting, so it is
    // the canonical case that used to return +6 instead of the correct -6.
    auto d1 = Matrix<float64>::fromRows({{4, 3},
                                         {6, 3}});
    auto d2 = Matrix<float64>::fromRows({{1, 2},
                                         {3, 4}});
    auto d3 = Matrix<float64>::fromRows({{2, 0},
                                         {0, 3}});
    auto d4 = Matrix<float64>::fromRows({{1, 2, 3},
                                         {4, 5, 6},
                                         {7, 8, 10}});

    double v1 = matsar::Math::det(d1)(0);
    double v2 = matsar::Math::det(d2)(0);
    double v3 = matsar::Math::det(d3)(0);
    double v4 = matsar::Math::det(d4)(0);
    std::printf("det([[4,3],[6,3]])              = %.10g  (want -6)\n", v1);
    std::printf("det([[1,2],[3,4]])              = %.10g  (want -2)\n", v2);
    std::printf("det([[2,0],[0,3]])              = %.10g  (want  6)\n", v3);
    std::printf("det([[1,2,3],[4,5,6],[7,8,10]]) = %.10g  (want -3)\n", v4);
    if (!approx(v1, -6.0)) { std::printf("FAIL: det case 1 (want -6)\n"); ok = false; }
    if (!approx(v2, -2.0)) { std::printf("FAIL: det case 2 (want -2)\n"); ok = false; }
    if (!approx(v3, 6.0))  { std::printf("FAIL: det case 3 (want  6)\n"); ok = false; }
    if (!approx(v4, -3.0)) { std::printf("FAIL: det case 4 (want -3)\n"); ok = false; }

    // ---- real inv round-trip (compute precision unchanged) -------------------
    auto A = Matrix<float64>::fromRows({{4, 1},
                                        {2, 3}});
    auto Ainv = matsar::Math::inv(A);
    auto Ione = mmul(A, Ainv);
    if (!approx(Ione(0, 0), 1.0) || !approx(Ione(1, 1), 1.0) ||
        !approx(Ione(0, 1), 0.0) || !approx(Ione(1, 0), 0.0)) {
        std::printf("FAIL: mmul(A, inv(A)) = [[%f,%f],[%f,%f]], want identity\n",
                    Ione(0, 0), Ione(0, 1), Ione(1, 0), Ione(1, 1));
        ok = false;
    }

    // ---- complex inv round-trip (unblocked by the Det + rank fixes) ----------
    // C = [[2, i],[i, 2]], det = 4 - i*i = 5 (non-singular).
    auto C = Matrix<complex128>::fromRows({{complex128(2, 0), complex128(0, 1)},
                                           {complex128(0, 1), complex128(2, 0)}});
    auto Cinv = matsar::Math::inv(C);
    auto Cid = mmul(C, Cinv);
    complex128 e00 = Cid(0, 0), e01 = Cid(0, 1), e10 = Cid(1, 0), e11 = Cid(1, 1);
    if (std::abs(e00 - complex128(1, 0)) > 1e-9 || std::abs(e11 - complex128(1, 0)) > 1e-9 ||
        std::abs(e01) > 1e-9 || std::abs(e10) > 1e-9) {
        std::printf("FAIL: mmul(C, inv(C)) = [[(%g,%g),(%g,%g)],[(%g,%g),(%g,%g)]], want complex identity\n",
                    e00.real(), e00.imag(), e01.real(), e01.imag(),
                    e10.real(), e10.imag(), e11.real(), e11.imag());
        ok = false;
    }

    // ---- pinv round-trip (Phase 5B: element-type threading only, compute unchanged) ----
    auto P = Matrix<float64>::fromRows({{4, 1},
                                        {2, 3}});
    auto Pinv = matsar::Math::pinv(P);
    auto Prt = mmul(mmul(P, Pinv), P);
    if (!approx(Prt(0, 0), P(0, 0), 1e-6) || !approx(Prt(0, 1), P(0, 1), 1e-6) ||
        !approx(Prt(1, 0), P(1, 0), 1e-6) || !approx(Prt(1, 1), P(1, 1), 1e-6)) {
        std::printf("FAIL: mmul(mmul(P,pinv(P)),P) = [[%f,%f],[%f,%f]], want P = [[%f,%f],[%f,%f]]\n",
                    Prt(0, 0), Prt(0, 1), Prt(1, 0), Prt(1, 1),
                    P(0, 0), P(0, 1), P(1, 0), P(1, 1));
        ok = false;
    }

    // ---- complex pinv correctness (conjugate-transpose fix) ------------------
    // NOTE: for a SQUARE invertible matrix, (X(A)*A)^-1 * X(A) collapses to A^-1
    // *algebraically* for ANY invertible "transpose-like" op X -- including a plain
    // (non-conjugated) transpose -- via (X(A)*A)^-1 = A^-1*X(A)^-1, so a
    // pinv(square-invertible) == inv(square-invertible) check does NOT discriminate
    // this bug (verified empirically: it still passes with the buggy plain-transpose
    // code). The property that DOES require the conjugate transpose is that
    // A*pinv(A) must be Hermitian for a rectangular (here: tall, full column rank)
    // complex A -- with a plain transpose this was observed to be off by ~0.69 max
    // abs deviation; with the conjugate-transpose fix it is 0 to machine precision.
    auto Cx = Matrix<complex128>::fromRows({{complex128(1, 1), complex128(0, 0)},
                                            {complex128(0, 0), complex128(1, 0)},
                                            {complex128(1, 0), complex128(1, -1)}});
    auto CxPinv = matsar::Math::pinv(Cx);
    auto CxCxp = mmul(Cx, CxPinv);
    auto CxCxpH = CxCxp.conj().T();
    double herm_diff = 0;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            herm_diff = std::max(herm_diff, std::abs(CxCxp(i, j) - CxCxpH(i, j)));
    std::printf("max |A*pinv(A) - (A*pinv(A))^H| = %g  (want ~0; complex pseudo-inverse Hermitian-projector check)\n",
                herm_diff);
    if (herm_diff > 1e-6) {
        std::printf("FAIL: A*pinv(A) is not Hermitian for complex tall Cx (max |diff| = %g)\n", herm_diff);
        ok = false;
    } else {
        std::printf("PASS: A*pinv(A) is Hermitian within 1e-6 -- complex pseudo-inverse correct\n");
    }

    // Moore-Penrose property: A * pinv(A) * A == A (sanity check; holds regardless of
    // conjugation for full column rank A, so it is not a discriminator by itself, but
    // it must still hold).
    auto CxRt = mmul(CxCxp, Cx);
    bool mp_ok = true;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 2; ++j)
            if (std::abs(CxRt(i, j) - Cx(i, j)) > 1e-6) mp_ok = false;
    if (!mp_ok) {
        std::printf("FAIL: mmul(mmul(Cx,pinv(Cx)),Cx) != Cx\n");
        ok = false;
    } else {
        std::printf("PASS: mmul(mmul(Cx,pinv(Cx)),Cx) == Cx within 1e-6 (Moore-Penrose property holds)\n");
    }

    // ---- rank of a rank-deficient matrix -------------------------------------
    auto R = Matrix<float64>::fromRows({{1, 2},
                                        {2, 4}});
    auto rk = matsar::Math::rank(R);
    if (rk != 1) {
        std::printf("FAIL: rank([[1,2],[2,4]]) = %llu, want 1\n", (unsigned long long) rk);
        ok = false;
    }

    if (!ok)
        return 1;
    std::printf("PASS\n");
    return 0;
}

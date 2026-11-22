// Copyright 2026 Zhexian Zhou
// SPDX-License-Identifier: Apache-2.0
// test_linalg_values.cpp -- hand-derived value tests for LAPACK-backed math/
// functions (needs MATSAR_WITH_BLAS). Expected eigenvalues are worked out by hand;
// eigenvectors are checked through the defining equations, not against the code.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include <MatSAR/math/LinearAlgebra.hpp>
#include <algorithm>
#include <complex>
#include <vector>

using namespace matsar;
using cd = std::complex<double>;

namespace {
    template<typename T>
    cd at(const Matrix<T> &m, msize_type i, msize_type j) { return cd(m(i, j)); }

    // max over j of |A v_j - lambda_j v_j|  (right eigenvectors: A*V = V*D)
    template<typename T>
    double right_residual(const Matrix<float64> &A, const Matrix<T> &V, const Matrix<T> &D) {
        const msize_type n = A.getShape()[0];
        double worst = 0;
        for (msize_type j = 0; j < n; ++j)
            for (msize_type r = 0; r < n; ++r) {
                cd av = 0;
                for (msize_type k = 0; k < n; ++k) av += at(A, r, k) * at(V, k, j);
                worst = std::max(worst, std::abs(av - at(D, j, j) * at(V, r, j)));
            }
        return worst;
    }

    // max over j of |w_j^H A - lambda_j w_j^H|  (left eigenvectors: W'*A = D*W')
    template<typename T>
    double left_residual(const Matrix<float64> &A, const Matrix<T> &W, const Matrix<T> &D) {
        const msize_type n = A.getShape()[0];
        double worst = 0;
        for (msize_type j = 0; j < n; ++j)
            for (msize_type c = 0; c < n; ++c) {
                cd wa = 0;
                for (msize_type r = 0; r < n; ++r) wa += std::conj(at(W, r, j)) * at(A, r, c);
                worst = std::max(worst, std::abs(wa - at(D, j, j) * std::conj(at(W, c, j))));
            }
        return worst;
    }

    // LAPACK normalizes each eigenvector to unit 2-norm; also rules out all-zero columns.
    template<typename T>
    void check_unit_columns(const Matrix<T> &V) {
        const msize_type n = V.getShape()[0];
        for (msize_type j = 0; j < n; ++j) {
            double s = 0;
            for (msize_type r = 0; r < n; ++r) s += std::norm(at(V, r, j));
            CAPTURE(j);
            CHECK(std::abs(std::sqrt(s) - 1.0) < 1e-12);
        }
    }

    template<typename T>
    std::vector<cd> eigenvalues_sorted(const Matrix<T> &D) {
        std::vector<cd> ev;
        for (msize_type j = 0; j < D.getShape()[0]; ++j) ev.push_back(at(D, j, j));
        std::sort(ev.begin(), ev.end(), [](cd a, cd b) {
            return a.real() != b.real() ? a.real() < b.real() : a.imag() < b.imag();
        });
        return ev;
    }
}

// Non-symmetric, so left and right eigenvectors differ: catches V/W being swapped.
TEST_CASE("eig · real eigenvalues: D, right vectors V and left vectors W") {
    Matrix<float64> A = {{2, 1}, {0, 3}};   // triangular: eigenvalues 2 and 3
    auto VDW = Math::eig(A);
    REQUIRE(VDW.size() == 3);
    const auto &V = VDW[0], &D = VDW[1], &W = VDW[2];
    auto ev = eigenvalues_sorted(D);
    CHECK(std::abs(ev[0] - cd(2, 0)) < 1e-12);
    CHECK(std::abs(ev[1] - cd(3, 0)) < 1e-12);
    CHECK(right_residual(A, V, D) < 1e-12);
    CHECK(left_residual(A, W, D) < 1e-12);
    check_unit_columns(V);
    check_unit_columns(W);
}

// [[1,2],[-2,1]]: (1-l)^2 + 4 = 0  ->  l = 1 +/- 2i. The imaginary parts used to be dropped.
TEST_CASE("eig · complex eigenvalues with a complex result type") {
    Matrix<float64> A = {{1, 2}, {-2, 1}};
    auto VDW = Math::eig<complex128>(A);
    REQUIRE(VDW.size() == 3);
    const auto &V = VDW[0], &D = VDW[1], &W = VDW[2];
    auto ev = eigenvalues_sorted(D);
    CHECK(std::abs(ev[0] - cd(1, -2)) < 1e-12);
    CHECK(std::abs(ev[1] - cd(1, 2)) < 1e-12);
    CHECK(right_residual(A, V, D) < 1e-12);
    CHECK(left_residual(A, W, D) < 1e-12);
    check_unit_columns(V);
    check_unit_columns(W);
}

// condeig and funm consume eig's V/D/W; they must keep their results when eig's layout is fixed.
// A = [[2,1],[0,3]]: right vectors v1=(1,0), v2=(1,1)/sqrt2; left vectors w1=(1,-1)/sqrt2, w2=(0,1).
// s_j = |w_j||v_j| / |w_j . v_j| = sqrt2 for both (MATLAB condeig gives [1.4142; 1.4142]).
TEST_CASE("condeig · eigenvalue condition numbers") {
    Matrix<float64> A = {{2, 1}, {0, 3}};
    auto VDs = Math::condeig(A);
    REQUIRE(VDs.size() == 3);
    const auto &s = VDs[2];
    REQUIRE(s.getSize() == 2);
    CHECK(std::abs(s(0) - std::sqrt(2.0)) < 1e-12);
    CHECK(std::abs(s(1) - std::sqrt(2.0)) < 1e-12);
}

// funm(A, f) = V f(D) V^-1. With f(l) = l^2 the result must equal A*A = [[4,5],[0,9]].
TEST_CASE("funm · applies the function through the eigendecomposition") {
    Matrix<float64> A = {{2, 1}, {0, 3}};
    auto square = [](Matrix<float64> &d, const string &) {
        for (auto &x: d) x = x * x;
    };
    auto F = Math::funm(A, square, "square");
    REQUIRE(F.getShape() == mshape_type{2, 2});
    CHECK(std::abs(F(0, 0) - 4) < 1e-12);
    CHECK(std::abs(F(0, 1) - 5) < 1e-12);
    CHECK(std::abs(F(1, 0) - 0) < 1e-12);
    CHECK(std::abs(F(1, 1) - 9) < 1e-12);
}

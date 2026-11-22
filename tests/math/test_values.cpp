// Copyright 2026 Zhexian Zhou
// SPDX-License-Identifier: Apache-2.0
// test_values.cpp -- hand-derived value tests for math/ functions outside the
// NumPy-golden harness. Each TEST_CASE pins a wrong-result bug that was fixed, with
// expected values worked out by hand (never computed by the code under test).
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include <MatSAR/math/LogicFunctions.hpp>
#include <MatSAR/math/Probability.hpp>
#include <MatSAR/math/DigitalSignal.hpp>
#include <MatSAR/math/MatrixOperation.hpp>
#include <vector>

using namespace matsar;

namespace {
    // Sample moments computed directly (not via Math::mean/var, which are under test elsewhere).
    template<typename dType>
    void moments(const Matrix<dType> &r, double &mean, double &var, double &lo, double &hi) {
        double s = 0, ss = 0;
        lo = hi = static_cast<double>(*r.begin());
        for (const auto &x: r) {
            const double v = static_cast<double>(x);
            s += v;
            ss += v * v;
            lo = v < lo ? v : lo;
            hi = v > hi ? v : hi;
        }
        const double n = static_cast<double>(r.getSize());
        mean = s / n;
        var = (ss - n * mean * mean) / (n - 1);
    }

    Matrix<int64> vec(std::vector<int64> v) { return Matrix<int64>(v); }   // 1-D, shape {n}

    template<typename dType>
    void check_vec(const Matrix<dType> &got, const std::vector<int64> &want) {
        REQUIRE(got.getSize() == static_cast<msize_type>(want.size()));
        for (std::size_t i = 0; i < want.size(); ++i) {
            CAPTURE(i);
            CHECK(got(i) == want[i]);
        }
    }
}

// all/any used to inspect only the first element of each reduced slice.
TEST_CASE("all / any · global reduce looks past the first element") {
    Matrix<int64> first_true = {1, 0};
    Matrix<int64> first_false = {0, 1};
    CHECK(Math::all(first_true, none)(0) == false);
    CHECK(Math::any(first_false, none)(0) == true);

    Matrix<bool> b_first_true = {true, false};
    Matrix<bool> b_first_false = {false, true};
    CHECK(Math::all(b_first_true, none)(0) == false);
    CHECK(Math::any(b_first_false, none)(0) == true);
}

TEST_CASE("all / any · reduce along a dimension looks past the first element") {
    // rows: [1 1]
    //       [0 1]
    Matrix<int64> m = {{1, 1}, {0, 1}};
    auto all_cols = Math::all(m, 0);   // reduce over rows -> one value per column
    CHECK(all_cols(0, 0) == false);    // column 0 = [1, 0]
    CHECK(all_cols(0, 1) == true);     // column 1 = [1, 1]

    // rows: [1 0]
    //       [0 0]
    Matrix<int64> n = {{1, 0}, {0, 0}};
    auto any_rows = Math::any(n, 1);   // reduce over columns -> one value per row
    CHECK(any_rows(0, 0) == true);     // row 0 = [1, 0]
    CHECK(any_rows(1, 0) == false);    // row 1 = [0, 0]

    // rows: [0 1]
    //       [0 0]
    Matrix<bool> b = {{false, true}, {false, false}};
    auto any_brows = Math::any(b, 1);
    CHECK(any_brows(0, 0) == true);    // row 0 = [false, true]
    CHECK(any_brows(1, 0) == false);
    auto all_bcols = Math::all(Matrix<bool>{{true, true}, {false, true}}, 0);
    CHECK(all_bcols(0, 0) == false);   // column 0 = [true, false]
    CHECK(all_bcols(0, 1) == true);
}

// randn used to return rand(): uniform on (0,1), mean 0.5, variance 1/12.
// Sample size 40000: std error of the mean is 0.005 and of the variance ~0.007, so the
// tolerances below are > 7 sigma (no flakes), while uniform output misses by far.
TEST_CASE("randn · draws from the standard normal distribution") {
    auto r = Math::randn({200, 200});
    REQUIRE(r.getSize() == 40000);
    double mean, var, lo, hi;
    moments(r, mean, var, lo, hi);
    CHECK(std::abs(mean - 0.0) < 0.05);
    CHECK(std::abs(var - 1.0) < 0.05);
    CHECK(lo < -2.0);   // P(all 40000 draws >= -2) is ~0 for N(0,1); impossible for uniform(0,1)
    CHECK(hi > 2.0);
}

// exprnd(mu) is documented (like exppdf/expstat) with mu = the MEAN, but passed mu to
// std::exponential_distribution, whose parameter is the RATE (1/mean): samples had mean 1/mu.
// mu = 2.5, 40000 draws: std error of the mean is 2.5/200 = 0.0125, tolerance 0.1 (8 sigma).
// The old behavior gives a mean of 0.4.
TEST_CASE("exprnd · mu is the mean of the exponential distribution") {
    auto r = Math::exprnd(2.5, {200, 200});
    REQUIRE(r.getSize() == 40000);
    double mean, var, lo, hi;
    moments(r, mean, var, lo, hi);
    CHECK(std::abs(mean - 2.5) < 0.1);
    CHECK(std::abs(var - 6.25) < 1.0);   // variance = mu^2; std error ~0.09
    CHECK(lo >= 0.0);
}

// conv(u, v, "same") returns |u| elements starting at offset floor(|v|/2) of the full
// convolution (MATLAB convention). It used |v|-2, which is only right for |v| = 3 or 4.
TEST_CASE("conv · documented MATLAB example (full / same / valid)") {
    auto u = vec({-1, 2, 3, -2, 0, 1, 2});
    auto v = vec({2, 4, -1, 1});
    check_vec(Math::conv(u, v), {-2, 0, 15, 5, -9, 7, 6, 7, -1, 2});   // hand-computed
    check_vec(Math::conv(u, v, "same"), {15, 5, -9, 7, 6, 7, -1});
    check_vec(Math::conv(u, v, "valid"), {5, -9, 7, 6});
}

TEST_CASE("conv · \"same\" is centred for every kernel length") {
    // |v| = 1: full = [3 6 9], offset 0
    check_vec(Math::conv(vec({1, 2, 3}), vec({3}), "same"), {3, 6, 9});
    // |v| = 2: full = [0 1 2 3], offset 1
    check_vec(Math::conv(vec({1, 2, 3}), vec({0, 1}), "same"), {1, 2, 3});
    // |v| = 5 (box filter): full = [1 3 6 10 15 20 18 15 11 6], offset 2
    check_vec(Math::conv(vec({1, 2, 3, 4, 5, 6}), vec({1, 1, 1, 1, 1}), "same"), {6, 10, 15, 20, 18, 15});
}

// permute: a valid reorder must still work once the repeated-dimension guard is live
// (guards against a guard that fires on every input). B = permute(A, {3,1,2}) has
// size(B) = {size(A,3), size(A,1), size(A,2)} and B(k,i,j) = A(i,j,k).
TEST_CASE("permute · valid 3-D reorder") {
    Matrix<int64> a(mshape_type{2, 3, 4}, int64(0));
    for (msize_type i = 0; i < 2; ++i)
        for (msize_type j = 0; j < 3; ++j)
            for (msize_type k = 0; k < 4; ++k)
                a(mshape_type{i, j, k}) = 100 * i + 10 * j + k;
    auto b = Math::permute(a, {3, 1, 2});
    REQUIRE(b.getShape() == mshape_type{4, 2, 3});
    CHECK(b(mshape_type{0, 0, 0}) == 0);
    CHECK(b(mshape_type{3, 1, 2}) == 123);   // = a(1,2,3)
    CHECK(b(mshape_type{2, 0, 1}) == 12);    // = a(0,1,2)
    CHECK(b(mshape_type{1, 1, 0}) == 101);   // = a(1,0,1)
}

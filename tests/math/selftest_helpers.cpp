// Copyright 2026 Zhexian Zhou
// SPDX-License-Identifier: Apache-2.0
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include <cmath>
#include <limits>
#include "approx.hpp"
#include "golden.hpp"
using namespace matsar_test;
// NOTE: float64/int64 are GLOBAL typedefs in MatSAR (util/TypeUtil.hpp), not matsar:: members,
// so they are already visible unqualified here; a `using matsar::float64;` would not compile.

TEST_CASE("approx_eq: basic rel+abs") {
    CHECK(approx_eq(1.0, 1.0, 0, 0));
    CHECK(approx_eq(1.0 + 1e-13, 1.0, 1e-12, 1e-15));
    CHECK_FALSE(approx_eq(1.0 + 1e-6, 1.0, 1e-12, 1e-15));
    CHECK(approx_eq(0.0, 0.0, 0, 0));                       // exact
    CHECK_FALSE(approx_eq(1e-9, 0.0, 1e-12, 1e-15));        // atol term must reject this
    CHECK(approx_eq(1e-16, 0.0, 0.0, 1e-15));              // atol term must accept this
}
TEST_CASE("approx_eq: NaN and Inf") {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    CHECK(approx_eq(nan, nan, 1e-12, 1e-15));               // NaN matches NaN
    CHECK_FALSE(approx_eq(1.0, nan, 1e-12, 1e-15));
    CHECK_FALSE(approx_eq(nan, 1.0, 1e-12, 1e-15));         // a function that unexpectedly returns NaN must FAIL
    CHECK(approx_eq(INFINITY, INFINITY, 1e-12, 1e-15));
    CHECK_FALSE(approx_eq(-INFINITY, INFINITY, 1e-12, 1e-15));
}
TEST_CASE("build<T>: 1-D and 2-D round-trip logical") {
    auto v = build<float64>({1., 2., 3.}, {3});
    CHECK(v.getShape().size() == 1);
    CHECK(v(0) == 1.); CHECK(v(2) == 3.);
    auto m = build<float64>({1., 2., 3., 4., 5., 6.}, {2, 3});
    CHECK((std::size_t)m.getShape()[0] == 2);
    CHECK((std::size_t)m.getShape()[1] == 3);
    CHECK(m(0, 1) == 2.);   // logical row-major: 2nd of first row
    CHECK(m(1, 0) == 4.);
    auto mi = build<int64>({1., 2., 3., 4.}, {4});   // double->int64 cast path
    CHECK(mi(0) == 1); CHECK(mi(3) == 4);
}
TEST_CASE("check_matrix: passes on exact (sanity)") {
    auto m = build<float64>({5., 7., 9.}, {1, 3});
    check_matrix("ok", m, {5., 7., 9.}, {1, 3}, 0, 0);   // must pass
}

// Copyright 2026 Zhexian Zhou
// SPDX-License-Identifier: Apache-2.0
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include <cmath>
#include <limits>
#include "approx.hpp"
using namespace matsar_test;

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

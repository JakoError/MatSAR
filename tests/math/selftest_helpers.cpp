#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include <limits>
#include "approx.hpp"
using namespace matsar_test;

TEST_CASE("approx_eq: basic rel+abs") {
    CHECK(approx_eq(1.0, 1.0, 0, 0));
    CHECK(approx_eq(1.0 + 1e-13, 1.0, 1e-12, 1e-15));
    CHECK_FALSE(approx_eq(1.0 + 1e-6, 1.0, 1e-12, 1e-15));
    CHECK(approx_eq(0.0, 0.0, 0, 0));                       // exact
}
TEST_CASE("approx_eq: NaN and Inf") {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    CHECK(approx_eq(nan, nan, 1e-12, 1e-15));               // NaN matches NaN
    CHECK_FALSE(approx_eq(1.0, nan, 1e-12, 1e-15));
    CHECK(approx_eq(INFINITY, INFINITY, 1e-12, 1e-15));
    CHECK_FALSE(approx_eq(-INFINITY, INFINITY, 1e-12, 1e-15));
}

// Copyright 2026 Zhexian Zhou
// SPDX-License-Identifier: Apache-2.0
// Scalar functions from NumberOperation.hpp (round, gcd, numden, factorial, nchoosek).
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <MatSAR/Matrix.hpp>
#include <MatSAR/NumberOperation.hpp>
#include <cmath>
#include <complex>
#include <limits>

using namespace matsar;

TEST_CASE("round: half away from zero, exact just below one half") {
    CHECK(matsar::round(0.5) == 1.0);
    CHECK(matsar::round(-0.5) == -1.0);
    CHECK(matsar::round(2.5) == 3.0);
    CHECK(matsar::round(-2.5) == -3.0);
    CHECK(matsar::round(0.49999999999999994) == 0.0);  // the MinGW-w64 C runtime round() gives 1
    CHECK(matsar::round(-0.49999999999999994) == 0.0);
    CHECK(matsar::round(4503599627370495.5) == 4503599627370496.0);
    CHECK(matsar::round(1e300) == 1e300);
    CHECK(std::signbit(matsar::round(-0.25)));  // -0.25 rounds to -0
    CHECK(std::isnan(matsar::round(NAN)));
    CHECK(matsar::round(-INFINITY) == -INFINITY);
    CHECK(matsar::round(complex128(1.5, -0.49999999999999994)) == complex128(2, 0));
}

TEST_CASE("gcd is never negative") {
    CHECK(matsar::gcd<int64>(12, 18) == 6);
    CHECK(matsar::gcd<int64>(4, -6) == 2);
    CHECK(matsar::gcd<int64>(-4, 6) == 2);
    CHECK(matsar::gcd<int64>(-4, -6) == 2);
    CHECK(matsar::gcd<int64>(-4, 0) == 4);
    CHECK(matsar::gcd<int64>(0, 0) == 0);
    CHECK(matsar::gcd<uint64>(12, 18) == 6);
}

TEST_CASE("numden reduces the fraction and keeps the sign on the numerator") {
    auto r = matsar::numden(int64(3), int64(-6));
    CHECK(r[0] == -1);
    CHECK(r[1] == 2);
    r = matsar::numden(int64(-4), int64(-6));
    CHECK(r[0] == 2);
    CHECK(r[1] == 3);
    r = matsar::numden(int64(0), int64(-5));
    CHECK(r[0] == 0);
    CHECK(r[1] == 1);
}

TEST_CASE("factorial") {
    CHECK(matsar::factorial(0) == 1.0);
    CHECK(matsar::factorial(5) == 120.0);
    CHECK(matsar::factorial(20) == 2432902008176640000.0);
}

TEST_CASE("nchoosek: exact while representable, no needless overflow") {
    CHECK(matsar::nchoosek(5, 2) == 10.0);
    CHECK(matsar::nchoosek(0, 0) == 1.0);
    CHECK(matsar::nchoosek(50, 25) == 126410606437752.0);  // was off by rounding
    CHECK(matsar::nchoosek(2000, 1999) == 2000.0);         // was Inf
    CHECK(matsar::nchoosek(1000, 500) == doctest::Approx(2.7028824094543656e299).epsilon(1e-13));
    CHECK(std::isnan(matsar::nchoosek(3, 5)));
    CHECK(std::isnan(matsar::nchoosek(2.5, 1)));
}

// TypeUtil::eps(x) is MATLAB's eps(x): the distance from |x| to the next larger number.
// It was x - nextafter(x, epsilon): half the spacing at a power of two, and negative at 0.
TEST_CASE("eps: spacing to the next larger number, as MATLAB") {
    using matsar::Util::TypeUtil::eps;
    CHECK(eps(1.0) == std::ldexp(1.0, -52));
    CHECK(eps(1.5) == std::ldexp(1.0, -52));
    CHECK(eps(2.0) == std::ldexp(1.0, -51));
    CHECK(eps(-2.0) == std::ldexp(1.0, -51));
    CHECK(eps(0.0) == std::numeric_limits<double>::denorm_min());
    CHECK(eps(1e-310) == std::numeric_limits<double>::denorm_min());
    CHECK(eps(std::numeric_limits<double>::max()) == std::ldexp(1.0, 971));
    CHECK(std::isnan(eps(std::numeric_limits<double>::infinity())));
    CHECK(std::isnan(eps(std::numeric_limits<double>::quiet_NaN())));
    CHECK(eps(1.0f) == std::ldexp(1.0f, -23));
    CHECK(eps(0.0f) == std::numeric_limits<float>::denorm_min());
}

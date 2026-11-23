// Copyright 2026 Zhexian Zhou
// SPDX-License-Identifier: Apache-2.0
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <MatSAR/Matrix.hpp>
#include <MatSAR/MatrixCreation.hpp>

using namespace matsar;

// diag(2,3,4) is layout-independent; a column-major-aware square product below.
TEST_CASE("mmul: diagonal * identity == diagonal") {
    Matrix<float64> m = {{2, 0, 0}, {0, 3, 0}, {0, 0, 4}};
    auto id = eye<float64>(3);
    auto p = mmul(m, id);
    CHECK(p(0, 0) == doctest::Approx(2));
    CHECK(p(1, 1) == doctest::Approx(3));
    CHECK(p(2, 2) == doctest::Approx(4));
    CHECK(p(0, 1) == doctest::Approx(0));
}

// operator% is an alias for mmul.
TEST_CASE("operator% == mmul") {
    Matrix<float64> m = {{2, 0, 0}, {0, 3, 0}, {0, 0, 4}};
    auto id = eye<float64>(3);
    auto p = m % id;
    CHECK(p(0, 0) == doctest::Approx(2));
    CHECK(p(2, 2) == doctest::Approx(4));
}

// A column-vector operand (one column) lost its second dimension when broadcast to its own
// shape, and mmul then read past the end of the shape: matrix * column vector, an outer
// product and 1x1 products overflowed the heap (AddressSanitizer: heap-buffer-overflow).
TEST_CASE("mmul: operands with a single column") {
    using V2 = std::vector<std::vector<float64>>;
    auto A = Matrix<float64>::fromRows(V2{{1, 2}, {3, 4}});
    auto x = Matrix<float64>::fromRows(V2{{5}, {6}});
    auto Ax = mmul(A, x);
    REQUIRE(Ax.getShape() == mshape_type{2, 1});
    CHECK(Ax(0, 0) == 17);
    CHECK(Ax(1, 0) == 39);

    auto r = Matrix<float64>::fromRows(V2{{1, 2}});
    auto rx = mmul(r, x);
    REQUIRE(rx.getShape() == mshape_type{1, 1});
    CHECK(rx(0, 0) == 17);

    auto c = Matrix<float64>::fromRows(V2{{1}, {2}, {3}});
    auto outer = mmul(c, Matrix<float64>::fromRows(V2{{4, 5}}));
    REQUIRE(outer.getShape() == mshape_type{3, 2});
    CHECK(outer(2, 0) == 12);
    CHECK(outer(1, 1) == 10);

    auto one = mmul(Matrix<float64>::fromRows(V2{{4}}), Matrix<float64>::fromRows(V2{{5}}));
    REQUIRE(one.getShape() == mshape_type{1, 1});
    CHECK(one(0, 0) == 20);
}

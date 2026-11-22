// Copyright 2026 Zhexian Zhou
// SPDX-License-Identifier: Apache-2.0
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <MatSAR/Matrix.hpp>

using namespace matsar;

// Uses explicit fromRows so the layout is fixed regardless of the
// MATSAR_LEGACY_COLUMN_INITIALIZER_LIST flag. fromRows({{1,2},{3,4}}) = rows
// [1,2],[3,4]; fromRows({{2,2}}) = one row -> shape {1,2}, broadcasts across rows
// to {2,2} (value 2 everywhere). Shapes differ -> the Matrix-Matrix broadcast
// branch is exercised.
TEST_CASE("operator< triggers the Matrix-Matrix broadcast branch") {
    auto a = Matrix<int64>::fromRows({{1, 2}, {3, 4}});   // a(0,0)=1 a(0,1)=2 a(1,0)=3 a(1,1)=4
    auto b = Matrix<int64>::fromRows({{2, 2}});            // shape {1,2}; compares against 2
    auto r = a < b;                                        // a(i,j) < 2
    CHECK(r(0, 0));                                         // 1 < 2
    CHECK_FALSE(r(0, 1));                                   // 2 < 2
    CHECK_FALSE(r(1, 0));                                   // 3 < 2
    CHECK_FALSE(r(1, 1));                                   // 4 < 2
}

TEST_CASE("other comparisons with broadcasting still correct") {
    auto a = Matrix<int64>::fromRows({{1, 2}, {3, 4}});
    auto b = Matrix<int64>::fromRows({{2, 2}});            // shape {1,2}, value 2
    CHECK((a <= b)(0, 1));                                  // 2 <= 2
    CHECK((a > b)(1, 0));                                  // 3 >  2
    CHECK((a >= b)(0, 1));                                  // 2 >= 2
    CHECK((a == b)(0, 1));                                  // 2 == 2
    CHECK((a != b)(0, 0));                                  // 1 != 2
}

// Comparisons are const member functions: they must work on a const Matrix (e.g. through a
// const Matrix& parameter), against a scalar of another type and against another Matrix.
TEST_CASE("comparisons work on a const Matrix") {
    const auto a = Matrix<float64>::fromRows({{1.0, 2.0}, {3.0, 4.0}});
    const auto b = Matrix<float64>::fromRows({{2.0, 2.0}});  // broadcast: compares against 2
    // every operator, in both the Matrix and the scalar form
    CHECK((a < b)(0, 0));
    CHECK((a < 2)(0, 0));
    CHECK((a <= b)(0, 1));
    CHECK((a <= 2)(0, 1));
    CHECK((a > b)(1, 0));
    CHECK((a > 2.5)(1, 0));
    CHECK((a >= b)(1, 1));
    CHECK_FALSE((a >= 3)(0, 1));
    CHECK((a == b)(0, 1));
    CHECK((a == 4)(1, 1));
    CHECK((a != b)(0, 0));
    CHECK((a != 0)(0, 0));  // int scalar against double elements
    CHECK_FALSE((a != a)(1, 0));
}

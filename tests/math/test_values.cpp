// Copyright 2026 Zhexian Zhou
// SPDX-License-Identifier: Apache-2.0
// test_values.cpp -- hand-derived value tests for math/ functions outside the
// NumPy-golden harness. Each TEST_CASE pins a wrong-result bug that was fixed, with
// expected values worked out by hand (never computed by the code under test).
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include <MatSAR/math/LogicFunctions.hpp>

using namespace matsar;

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

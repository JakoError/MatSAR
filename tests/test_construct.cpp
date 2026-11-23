// Copyright 2026 Zhexian Zhou
// SPDX-License-Identifier: Apache-2.0
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <vector>
#include <MatSAR/Matrix.hpp>

using namespace matsar;

TEST_CASE("fromColumns: inner brace = column (legacy layout)") {
    auto c = Matrix<int64>::fromColumns({{1, 2, 3}, {4, 5, 6}});  // columns [1,2,3],[4,5,6]
    CHECK(c.getShape()[0] == 3);   // 3 rows
    CHECK(c.getShape()[1] == 2);   // 2 cols
    CHECK(c(0, 0) == 1);
    CHECK(c(1, 0) == 2);
    CHECK(c(2, 0) == 3);
    CHECK(c(0, 1) == 4);
}

TEST_CASE("fromRows: inner brace = row") {
    auto r = Matrix<int64>::fromRows({{1, 2, 3}, {4, 5, 6}});     // rows [1,2,3],[4,5,6]
    CHECK(r.getShape()[0] == 2);   // 2 rows
    CHECK(r.getShape()[1] == 3);   // 3 cols
    CHECK(r(0, 0) == 1);
    CHECK(r(0, 1) == 2);
    CHECK(r(0, 2) == 3);
    CHECK(r(1, 0) == 4);
}

TEST_CASE("fromRows equals fromColumns transposed") {
    auto c = Matrix<int64>::fromColumns({{1, 2, 3}, {4, 5, 6}});
    auto r = Matrix<int64>::fromRows({{1, 2, 3}, {4, 5, 6}});
    auto ct = c.T();
    CHECK(r.getShape()[0] == ct.getShape()[0]);
    CHECK(r.getShape()[1] == ct.getShape()[1]);
    CHECK(r(0, 2) == ct(0, 2));
    CHECK(r(1, 0) == ct(1, 0));
}

TEST_CASE("default nested initializer list is row-wise (unless legacy flag)") {
    Matrix<int64> m = {{1, 2, 3}, {4, 5, 6}};
#ifdef MATSAR_LEGACY_COLUMN_INITIALIZER_LIST
    CHECK(m.getShape()[0] == 3);   // legacy: column-wise -> 3x2
    CHECK(m.getShape()[1] == 2);
    CHECK(m(0, 1) == 4);
#else
    CHECK(m.getShape()[0] == 2);   // default: row-wise -> 2x3
    CHECK(m.getShape()[1] == 3);
    CHECK(m(0, 1) == 2);
    CHECK(m(1, 0) == 4);
#endif
}

TEST_CASE("flat 1-D initializer list is unchanged (shape {N})") {
    Matrix<int64> v = {1, 2, 3};
    CHECK(v.getSize() == 3);
    CHECK(v.getShape()[0] == 3);
    CHECK(v(0) == 1);
    CHECK(v(2) == 3);
}

TEST_CASE("vector<vector> constructor matches the row-wise default") {
    std::vector<std::vector<int64>> vv = {{1, 2, 3}, {4, 5, 6}};
    Matrix<int64> m(vv);
#ifdef MATSAR_LEGACY_COLUMN_INITIALIZER_LIST
    CHECK(m.getShape()[0] == 3);   // legacy column-wise -> 3x2
    CHECK(m.getShape()[1] == 2);
    CHECK(m(0, 1) == 4);
#else
    CHECK(m.getShape()[0] == 2);   // row-wise -> 2x3 (consistent with the init-list)
    CHECK(m.getShape()[1] == 3);
    CHECK(m(0, 1) == 2);
    CHECK(m(1, 0) == 4);
#endif
}

// A literal 0 is also a null pointer constant, so it must not pick Matrix(shape, dType *).
TEST_CASE_TEMPLATE("Matrix(shape, 0) fills with zeros", T, float64, float32, int64, int32, bool, complex128) {
    Matrix<T> m(mshape_type{2, 3}, 0);
    REQUIRE(m.getShape() == mshape_type{2, 3});
    for (msize_type i = 0; i < m.getSize(); ++i)
        CHECK(m(i) == T(0));
    Matrix<T> b({3}, 0);
    REQUIRE(b.getShape() == mshape_type{3});
    CHECK(b(2) == T(0));
}

TEST_CASE("Matrix(shape, pointer) copies the buffer; a null pointer gives zeros") {
    float64 buf[] = {1, 2, 3, 4};
    Matrix<float64> m(mshape_type{2, 2}, buf);
    CHECK(m(0) == 1);
    CHECK(m(3) == 4);
    float64 *none = nullptr;
    Matrix<float64> z(mshape_type{2, 2}, none);
    CHECK(z(0) == 0);
    CHECK(z(3) == 0);
}

TEST_CASE("an empty nested vector gives an empty matrix") {
    Matrix<float64> none(std::vector<std::vector<float64>>{});
    CHECK(none.getSize() == 0);
    CHECK(none.getShape() == mshape_type{0});
    Matrix<float64> rows(std::vector<std::vector<float64>>{{}, {}});
    CHECK(rows.getSize() == 0);
    CHECK(rows.getShape() == mshape_type{0});
    auto cols = Matrix<float64>::fromColumns(std::vector<std::vector<float64>>{});
    CHECK(cols.getSize() == 0);
    Matrix<float64> deep(std::vector<std::vector<std::vector<float64>>>{});
    CHECK(deep.getSize() == 0);
    CHECK(deep.getShape() == mshape_type{0});
}

TEST_CASE("fromRows / fromColumns accept vector<vector>") {
    std::vector<std::vector<int64>> vv = {{1, 2, 3}, {4, 5, 6}};
    auto r = Matrix<int64>::fromRows(vv);
    auto c = Matrix<int64>::fromColumns(vv);
    CHECK(r.getShape()[0] == 2);   // rows -> 2x3
    CHECK(r.getShape()[1] == 3);
    CHECK(r(1, 0) == 4);
    CHECK(c.getShape()[0] == 3);   // columns -> 3x2
    CHECK(c.getShape()[1] == 2);
    CHECK(c(0, 1) == 4);
}

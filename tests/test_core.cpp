// Copyright 2026 Zhexian Zhou
// SPDX-License-Identifier: Apache-2.0
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <MatSAR/Matrix.hpp>
#include <MatSAR/MatrixCreation.hpp>

using namespace matsar;

TEST_CASE("core container constructs and indexes") {
    Matrix<int64> a = {{1, 2}, {3, 4}};      // 2x2, row-major
    CHECK(a.getShape()[0] == 2);
    CHECK(a.getShape()[1] == 2);
    CHECK(a(0, 0) == 1);
    CHECK(a(1, 1) == 4);
}

TEST_CASE("creation helpers") {
    auto e = eye<int64>(3);
    CHECK(e(0, 0) == 1);
    CHECK(e(0, 1) == 0);
    CHECK(e(2, 2) == 1);
}

TEST_CASE("isDiag: square, off-diagonal elements all zero") {
    Matrix<float64> d = {{1.0, 0.0}, {0.0, 2.0}};
    Matrix<float64> offd = {{1.0, 3.0}, {0.0, 2.0}};
    Matrix<float64> lower = {{1.0, 0.0}, {4.0, 2.0}};
    Matrix<float64> wide = {{1.0, 0.0, 0.0}, {0.0, 2.0, 0.0}};
    CHECK(d.isDiag());
    CHECK_FALSE(offd.isDiag());
    CHECK_FALSE(lower.isDiag());
    CHECK_FALSE(wide.isDiag());  // only square matrices are diagonal
    CHECK(eye<int64>(3).isDiag());
    CHECK(Matrix<int64>({7}).isDiag());  // 1x1
    CHECK(Matrix<int64>().isDiag());     // empty is square, so diagonal
    CHECK_FALSE(Matrix<int64>({1, 2}).isDiag());
}

// "Return a copy" written as `return {m}` list-initializes through the concatenating
// initializer_list<Matrix> constructor, and cat() squeezes a trailing singleton dimension:
// a 3x1 column came back with shape {3}. Each of these must keep the shape.
TEST_CASE("copies keep a trailing singleton dimension") {
    const Matrix<float64> col(mshape_type{3, 1}, 2.0);
    const mshape_type s{3, 1};
    CHECK(col.reshape(s).getShape() == s);
    CHECK(col.broadcast(s).getShape() == s);
    CHECK((+col).getShape() == s);
    CHECK(col.squeeze().getShape() == s);   // a 2-D matrix is left as it is
    CHECK(flip(col, 3).getShape() == s);    // a dim beyond the dimensions: the matrix itself
    CHECK(rot90(col, 0).getShape() == s);
    CHECK(rot90(col, 4).getShape() == s);
    const Matrix<float64> one(mshape_type{1, 1}, 5.0);
    CHECK(one.broadcast(mshape_type{1, 1}).getShape() == mshape_type{1, 1});
}

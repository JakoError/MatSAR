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

// A 1-D {n} matrix stands for an n x 1 column (as in m(i, j) and isDiag), so its transpose is
// the 1 x n row. T() used to swap shape[0] with a shape[1] that a 1-D shape does not have.
TEST_CASE("T() of a 1-D vector is a 1xN row") {
    const Matrix<float64> v = {1.0, 2.0, 3.0};
    const auto t = v.T();
    REQUIRE(t.getShape() == mshape_type{1, 3});
    CHECK(t(0, 0) == 1.0);
    CHECK(t(0, 2) == 3.0);
    const auto col = t.T();
    REQUIRE(col.getShape() == mshape_type{3, 1});
    CHECK(col(2, 0) == 3.0);
    CHECK(Matrix<int64>({7}).T().getShape() == mshape_type{1, 1});
}

TEST_CASE("T() of an empty matrix is empty") {
    CHECK(Matrix<float64>().T().isEmpty());
    CHECK(Matrix<float64>(mshape_type{0}, 1.0).T().isEmpty());
}

TEST_CASE("T() swaps the first two axes of every page") {
    const auto a = Matrix<int64>::fromRows({{1, 2, 3}, {4, 5, 6}});   // 2x3 under the legacy flag too
    const auto t = a.T();
    REQUIRE(t.getShape() == mshape_type{3, 2});
    CHECK(t(2, 1) == 6);
    CHECK(t(0, 1) == 4);
    Matrix<int64> p(mshape_type{2, 3, 2}, 0);
    for (msize_type i = 0; i < p.getSize(); ++i)
        p(i) = i;
    const auto tp = p.T();
    REQUIRE(tp.getShape() == mshape_type{3, 2, 2});
    for (msize_type n = 0; n < 2; ++n)
        for (msize_type i = 0; i < 2; ++i)
            for (msize_type j = 0; j < 3; ++j)
                CHECK(tp(n, j, i) == p(n, i, j));
}

TEST_CASE("rot90 of a 1-D vector rotates the column it stands for, as MATLAB") {
    const Matrix<float64> v = {1.0, 2.0, 3.0};
    const auto r1 = rot90(v);   // [1 2 3]
    REQUIRE(r1.getShape() == mshape_type{1, 3});
    CHECK(r1(0, 0) == 1.0);
    CHECK(r1(0, 2) == 3.0);
    const auto r2 = rot90(v, 2);   // [3; 2; 1]
    REQUIRE(r2.getShape() == mshape_type{3, 1});
    CHECK(r2(0, 0) == 3.0);
    CHECK(r2(2, 0) == 1.0);
    const auto r3 = rot90(v, 3);   // [3 2 1]
    REQUIRE(r3.getShape() == mshape_type{1, 3});
    CHECK(r3(0, 0) == 3.0);
    CHECK(r3(0, 2) == 1.0);
}

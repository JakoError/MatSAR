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

// Each pair of braces is one dimension. With one initializer_list overload per depth, inner
// braces holding one element could also initialize a scalar, so such literals were ambiguous.
namespace {
    template<class T>
    bool same(const Matrix<T> &a, const Matrix<T> &b) {
        if (a.getShape() != b.getShape()) return false;
        for (msize_type i = 0; i < a.getSize(); ++i)
            if (!(a(i) == b(i))) return false;
        return true;
    }

    // The 2-D layout the literal must have: row-wise, or column-wise under the legacy flag.
    template<class T>
    Matrix<T> layout2d(std::initializer_list<std::initializer_list<T>> rows) {
#ifdef MATSAR_LEGACY_COLUMN_INITIALIZER_LIST
        return Matrix<T>::fromColumns(rows);
#else
        return Matrix<T>::fromRows(rows);
#endif
    }
}

TEST_CASE("rows of one element make a column, not a 1-D list") {
    Matrix<float64> c = {{1.0}, {2.0}, {3.0}};
#ifndef MATSAR_LEGACY_COLUMN_INITIALIZER_LIST
    CHECK(c.getShape() == mshape_type{3, 1});
    CHECK(c(2, 0) == 3.0);
#endif
    CHECK(same(c, layout2d<float64>({{1.0}, {2.0}, {3.0}})));
    Matrix<int64> i = {{10}, {20}};
    CHECK(same(i, layout2d<int64>({{10}, {20}})));
    Matrix<float64> mixed = {{1}, {2.5}};   // an int and a floating literal
    CHECK(same(mixed, layout2d<float64>({{1.0}, {2.5}})));
}

TEST_CASE("{{x}} is a 1x1 matrix and {x} a 1-D one") {
    Matrix<float64> one = {{2.0}};
    CHECK(one.getShape() == mshape_type{1, 1});
    CHECK(one(0, 0) == 2.0);
    Matrix<float64> flat = {2.0};
    CHECK(flat.getShape() == mshape_type{1});
}

TEST_CASE("a literal 0 and mixed literals inside nested braces") {
    Matrix<int64> a = {{1, 0}, {0, 2}};
    CHECK(same(a, layout2d<int64>({{1, 0}, {0, 2}})));
    Matrix<float64> b = {{1, 0.5}, {0, 2}};
    CHECK(same(b, layout2d<float64>({{1.0, 0.5}, {0.0, 2.0}})));
}

TEST_CASE("complex literals mix real and complex elements") {
    Matrix<complex128> z = {{complex128(1, 2)}, {3.0}};
    CHECK(same(z, layout2d<complex128>({{complex128(1, 2)}, {complex128(3, 0)}})));
    Matrix<complex128> w = {1.0, complex128(0, 1)};
    CHECK(w.getShape() == mshape_type{2});
    CHECK(w(1) == complex128(0, 1));
}

TEST_CASE("three or more levels of braces keep the legacy N-D order") {
    // The innermost braces run along the first dimension: m({i, j, k}) = literal[k][j][i].
    Matrix<int64> m = {{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}};
    REQUIRE(m.getShape() == mshape_type{2, 2, 2});
    CHECK(m({1, 0, 0}) == 2);
    CHECK(m({0, 1, 0}) == 3);
    CHECK(m({0, 0, 1}) == 5);
    Matrix<int64> r = {{{1, 2, 3}, {4, 5, 6}}};
    REQUIRE(r.getShape() == mshape_type{3, 2, 1});
    CHECK(r({2, 1, 0}) == 6);
    Matrix<int64> t = {{{1}, {2}}};   // one element innermost
    REQUIRE(t.getShape() == mshape_type{1, 2, 1});
    CHECK(t({0, 1, 0}) == 2);
    Matrix<int64> u = {{{7}}};
    CHECK(u.getShape() == mshape_type{1, 1, 1});
    Matrix<int64> q = {{{{1}, {2}}}};
    REQUIRE(q.getShape() == mshape_type{1, 2, 1, 1});
    CHECK(q({0, 1, 0, 0}) == 2);
}

TEST_CASE("empty braces give an empty matrix") {
    Matrix<float64> e = {{}};
    CHECK(e.getSize() == 0);
    CHECK(e.getShape() == mshape_type{0});   // as Matrix(shape, v) and the vector constructors
    Matrix<float64> rows = {{}, {}};
    CHECK(rows.getSize() == 0);
    CHECK(rows.getShape() == mshape_type{0});
}

TEST_CASE("an initializer_list object is read like the literal") {
    std::initializer_list<std::initializer_list<float64>> rows = {{1.0}, {2.0}};
    Matrix<float64> m(rows);
    CHECK(same(m, layout2d<float64>({{1.0}, {2.0}})));
    std::initializer_list<float64> flat = {1.0, 2.0, 3.0};
    Matrix<float64> v(flat);
    CHECK(v.getShape() == mshape_type{3});
    CHECK(v(2) == 3.0);
}

// An element keeps its kind: a floating value cannot become an integer, nor a complex a real.
static_assert(std::is_constructible_v<NestedList<float64>, int>);
static_assert(std::is_constructible_v<NestedList<float32>, float64>);
static_assert(std::is_constructible_v<NestedList<complex128>, float64>);
static_assert(std::is_constructible_v<NestedList<int64>, int>);
static_assert(!std::is_constructible_v<NestedList<int64>, float64>);
static_assert(!std::is_constructible_v<NestedList<float64>, complex128>);

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

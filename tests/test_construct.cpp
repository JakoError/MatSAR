#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <Matrix.hpp>

using namespace MatSAR;

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

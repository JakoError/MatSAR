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

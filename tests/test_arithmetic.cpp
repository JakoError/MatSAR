#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <Matrix.hpp>
#include <MatrixCreation.hpp>

using namespace MatSAR;

// Uses explicit fromRows so the layout is fixed regardless of the
// MATSAR_LEGACY_COLUMN_INITIALIZER_LIST flag. fromRows({{1,2},{3,4}}) has rows
// [1,2],[3,4]: a(0,0)=1 a(0,1)=2 a(1,0)=3 a(1,1)=4.
// (The bare {{...}} default == fromRows is verified in test_construct.)

TEST_CASE("operator* is element-wise (NumPy semantics)") {
    auto a = Matrix<int64>::fromRows({{1, 2}, {3, 4}});   // rows [1,2],[3,4]
    auto b = Matrix<int64>::fromRows({{5, 6}, {7, 8}});   // rows [5,6],[7,8]
    auto c = a * b;                                        // element-wise
    CHECK(c(0, 0) == 5);                                   // 1*5
    CHECK(c(0, 1) == 12);                                  // 2*6
    CHECK(c(1, 0) == 21);                                  // 3*7
    CHECK(c(1, 1) == 32);                                  // 4*8
}

TEST_CASE("operator* broadcasts a row vector") {
    auto a = Matrix<int64>::fromRows({{1, 2}, {3, 4}});   // rows [1,2],[3,4]
    auto row = Matrix<int64>::fromRows({{10, 20}});        // one row -> shape {1,2}
    auto c = a * row;                                      // broadcast across rows: c(i,j)=a(i,j)*row(0,j)
    CHECK(c(0, 0) == 10);                                  // 1*10
    CHECK(c(0, 1) == 40);                                  // 2*20
    CHECK(c(1, 0) == 30);                                  // 3*10
    CHECK(c(1, 1) == 80);                                  // 4*20
}

TEST_CASE("scalar * matrix and matrix * scalar") {
    auto a = Matrix<int64>::fromRows({{1, 2}, {3, 4}});
    auto l = 3 * a;
    auto r = a * 3;
    CHECK(l(1, 1) == 12);                                  // 4*3
    CHECK(r(1, 1) == 12);
    CHECK(l(1, 0) == 9);                                   // 3*3  (a(1,0)==3 row-wise)
}

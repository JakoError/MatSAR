#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <Matrix.hpp>
#include <MatrixCreation.hpp>

using namespace MatSAR;

// MatSAR convention (this pass): nested initializer-list inner braces are
// COLUMNS (column-major). So {{1,2},{3,4}} has columns [1,2] and [3,4]:
//   a(0,0)=1 a(0,1)=3
//   a(1,0)=2 a(1,1)=4
// (Row-wise init + fromRows()/fromColumns() are a separate future design.)

TEST_CASE("operator* is element-wise (NumPy semantics, column-major layout)") {
    Matrix<int64> a = {{1, 2}, {3, 4}};   // cols [1,2],[3,4]
    Matrix<int64> b = {{5, 6}, {7, 8}};   // cols [5,6],[7,8]
    auto c = a * b;                        // element-wise, NOT matrix product
    CHECK(c(0, 0) == 5);                   // a(0,0)*b(0,0) = 1*5
    CHECK(c(1, 0) == 12);                  // a(1,0)*b(1,0) = 2*6
    CHECK(c(0, 1) == 21);                  // a(0,1)*b(0,1) = 3*7
    CHECK(c(1, 1) == 32);                  // a(1,1)*b(1,1) = 4*8
}

TEST_CASE("operator* broadcasts a column vector") {
    Matrix<int64> a = {{1, 2}, {3, 4}};   // a(0,1)==3, a(1,0)==2
    Matrix<int64> col = {{10, 20}};        // one inner brace = a column -> shape {2,1}
    auto c = a * col;                      // broadcast across columns: c(i,j)=a(i,j)*col(i)
    CHECK(c(0, 0) == 10);                  // 1*10
    CHECK(c(1, 0) == 40);                  // 2*20
    CHECK(c(0, 1) == 30);                  // 3*10
    CHECK(c(1, 1) == 80);                  // 4*20
}

TEST_CASE("scalar * matrix and matrix * scalar") {
    Matrix<int64> a = {{1, 2}, {3, 4}};
    auto l = 3 * a;
    auto r = a * 3;
    CHECK(l(1, 1) == 12);                  // 4*3
    CHECK(r(1, 1) == 12);
    CHECK(l(1, 0) == 6);                   // 2*3
}

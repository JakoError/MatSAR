#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <Matrix.hpp>

using namespace MatSAR;

// Column-major convention: {{1,2},{3,4}} has cols [1,2],[3,4] (a(0,1)==3);
// {{2,2}} is one column -> shape {2,1}, broadcasts across columns to {2,2}.
TEST_CASE("operator< triggers the Matrix-Matrix broadcast branch") {
    Matrix<int64> a = {{1, 2}, {3, 4}};   // a(0,0)=1 a(1,0)=2 a(0,1)=3 a(1,1)=4
    Matrix<int64> b = {{2, 2}};            // shape {2,1}; compares against 2
    auto r = a < b;                        // a(i,j) < 2
    CHECK(r(0, 0));                         // 1 < 2
    CHECK_FALSE(r(1, 0));                   // 2 < 2
    CHECK_FALSE(r(0, 1));                   // 3 < 2
    CHECK_FALSE(r(1, 1));                   // 4 < 2
}

TEST_CASE("other comparisons with broadcasting still correct") {
    Matrix<int64> a = {{1, 2}, {3, 4}};
    Matrix<int64> b = {{2, 2}};            // shape {2,1}, value 2
    CHECK((a <= b)(1, 0));                  // 2 <= 2
    CHECK((a >  b)(0, 1));                  // 3 >  2
    CHECK((a >= b)(1, 0));                  // 2 >= 2
    CHECK((a == b)(1, 0));                  // 2 == 2
    CHECK((a != b)(0, 0));                  // 1 != 2
}
